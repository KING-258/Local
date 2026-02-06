#!/usr/bin/env python3
"""
VisionM regression test runner (Web platform + Local backend endpoints).

- Targets the Node backend HTTP API (no UI automation here).
- Designed to cover the test IDs I outlined (AT_001–AT_0xx, etc.).
- Extend the `TEST_CASES` list and add more `test_*` functions to
  cover all 158 scenarios.

Pre‑requisites:
- Backend running on http://127.0.0.1:3000
- Python 3.9+
- pip install requests
- Any dataset/model fixtures you reference must exist under ./fixtures
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

BASE_API_URL = os.getenv("VISIONM_API_BASE_URL", "http://127.0.0.1:3000/api").rstrip("/")


# ---------------------------------------------------------------------------
# Generic test runner infrastructure
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TestContext:
    """Holds shared state across tests (auth tokens, IDs, etc.)."""

    api_base: str = BASE_API_URL
    # Optionally store auth token if you wire login flows here
    jwt: Optional[str] = None
    # You can store created IDs for reuse across tests
    dataset_ids: Dict[str, str] = dataclasses.field(default_factory=dict)
    inference_ids: Dict[str, str] = dataclasses.field(default_factory=dict)
    project_ids: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TestCase:
    id: str
    description: str
    fn: Callable[[TestContext], None]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class SkipTest(Exception):
    """Raised by tests when a precondition (fixture, ID, etc.) is missing."""
    pass


def _headers(ctx: TestContext, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build default headers for backend API.

    The backend's authMiddleware requires X-User-* headers. For automated tests
    we send a synthetic "platform_admin" user by default so routes pass auth
    checks without needing a real Supabase session.
    """
    h: Dict[str, str] = {
        "Content-Type": "application/json",
        # Synthetic user identity for tests – matches VALID_ROLES
        "X-User-Id": os.getenv("VISIONM_TEST_USER_ID", "test-runner"),
        "X-User-Role": os.getenv("VISIONM_TEST_USER_ROLE", "platform_admin"),
        "X-User-Company": os.getenv("VISIONM_TEST_COMPANY", "TestCompany"),
        "X-User-Email": os.getenv("VISIONM_TEST_EMAIL", "test@company.com"),
        "X-User-Company-Id": os.getenv("VISIONM_TEST_COMPANY_ID", "TEST_COMPANY_ID"),
    }
    if ctx.jwt:
        h["Authorization"] = f"Bearer {ctx.jwt}"
    if extra:
        h.update(extra)
    return h


def api_post(ctx: TestContext, path: str, json_body: dict,
             expected_status: int = 200) -> requests.Response:
    url = f"{ctx.api_base}{path}"
    resp = requests.post(url, headers=_headers(ctx), json=json_body, timeout=30)
    if resp.status_code != expected_status:
        raise AssertionError(
            f"POST {path} expected {expected_status}, got {resp.status_code}: {resp.text}"
        )
    return resp


def api_get(ctx: TestContext, path: str, params: Optional[dict] = None,
            expected_status: int = 200) -> requests.Response:
    url = f"{ctx.api_base}{path}"
    resp = requests.get(url, headers=_headers(ctx), params=params, timeout=30)
    if resp.status_code != expected_status:
        raise AssertionError(
            f"GET {path} expected {expected_status}, got {resp.status_code}: {resp.text}"
        )
    return resp


# ---------------------------------------------------------------------------
# Backend fixture discovery (projects, datasets, models)
# ---------------------------------------------------------------------------


def ensure_inference_fixtures(ctx: TestContext) -> None:
    """Populate ctx with a primary company/project, datasets, and model IDs.

    This connects to the running backend and discovers:
      - A primary (company, project) pair with at least one dataset & model
      - A dataset with testCount > 0 (for normal tests)
      - A dataset with testCount == 0 (for zero-test-images tests, if any)
      - A trained model for that (company, project)

    Raises RuntimeError if no suitable project exists at all.
    """
    # If already discovered, do nothing
    if "primary_project" in ctx.project_ids:
        return

    company = os.getenv("VISIONM_TEST_COMPANY", "RuzareInfoTech")

    # 1) Discover projects for this company via dashboard API
    resp = api_get(ctx, "/dashboard/projects", params={"company": company})
    data = resp.json()
    projects = data.get("projects") or []
    if not projects:
        raise RuntimeError(
            f"No projects found for company '{company}'. "
            "Set VISIONM_TEST_COMPANY to a company that has projects and data."
        )

    primary_project = (
        projects[0].get("project")
        or projects[0].get("name")
        or str(projects[0])
    )

    ctx.project_ids["primary_company"] = company
    ctx.project_ids["primary_project"] = primary_project

    # 2) Discover datasets for this (company, project)
    ds_resp = api_get(
        ctx,
        "/inference/datasets",
        params={"company": company, "project": primary_project},
    )
    ds_body = ds_resp.json()
    raw_datasets = ds_body.get("datasets") or ds_body.get("data") or ds_body
    if not isinstance(raw_datasets, list):
        raw_datasets = []

    for d in raw_datasets:
        if not isinstance(d, dict):
            continue
        did = d.get("datasetId") or d.get("_id") or d.get("id")
        if not did:
            continue
        test_count = d.get("testCount") or 0
        if test_count > 0 and "has_test_dataset" not in ctx.dataset_ids:
            ctx.dataset_ids["has_test_dataset"] = did
        if test_count == 0 and "zero_test_dataset" not in ctx.dataset_ids:
            ctx.dataset_ids["zero_test_dataset"] = did

    # 3) Discover models for this (company, project)
    models_resp = api_get(
        ctx,
        "/inference/models",
        params={"company": company, "project": primary_project},
    )
    m_body = models_resp.json()
    raw_models = m_body.get("models") or m_body.get("data") or m_body
    if not isinstance(raw_models, list):
        raw_models = []

    for m in raw_models:
        if not isinstance(m, dict):
            continue
        mid = m.get("modelId") or m.get("_id")
        if mid:
            ctx.dataset_ids["primary_model"] = mid
            break

    # Sanity check: we at least need one model and one dataset with tests
    if "primary_model" not in ctx.dataset_ids or "has_test_dataset" not in ctx.dataset_ids:
        raise RuntimeError(
            "Could not find both a trained model and a dataset with test images "
            f"for company='{company}', project='{primary_project}'. "
            "Run at least one training job first, then rerun this script."
        )


# ---------------------------------------------------------------------------
# Dataset validation tests (examples of AT_001–AT_018)
# ---------------------------------------------------------------------------


def test_AT_001_dataset_orphan_labels(ctx: TestContext) -> None:
    """AT_001 – Dataset validation – label/image one‑to‑one mapping.

    Upload a labelled dataset where label files exist without matching images.
    Expect: backend rejects with validation error mentioning orphan label files.

    Implementation assumes POST /api/dataset/upload accepts a ZIP file path in JSON
    (adjust to your real upload API).
    """
    # TODO: create this fixture ZIP in your repo:
    # ./fixtures/datasets/AT_001_orphan_labels.zip
    dataset_zip = Path("fixtures/datasets/AT_001_orphan_labels.zip")
    if not dataset_zip.exists():
        raise SkipTest(f"Fixture not found: {dataset_zip}")

    # Example API pattern – adjust to your real API contract:
    files = {"file": (dataset_zip.name, dataset_zip.read_bytes(), "application/zip")}
    url = f"{ctx.api_base}/dataset/upload"
    headers = _headers(ctx).copy()
    # Let requests set proper multipart boundary
    headers.pop("Content-Type", None)
    resp = requests.post(url, headers=headers, files=files, timeout=60)

    # We expect a 400 with a helpful message, not a 500.
    if resp.status_code != 400:
        raise AssertionError(
            f"AT_001 expected 400, got {resp.status_code}: {resp.text}"
        )

    body = resp.json()
    msg = (body.get("error") or "") + " " + (body.get("message") or "")
    if "orphan" not in msg.lower() and "label" not in msg.lower():
        raise AssertionError(
            f"AT_001 expected orphan/label validation error, got: {body}"
        )


def test_AT_008_dataset_invalid_label_format(ctx: TestContext) -> None:
    """
    AT_008 – Dataset validation – invalid label file format.

    Fixture contains YOLO label files with malformed lines.
    Expect: 400 with message referencing malformed / invalid labels.
    """
    dataset_zip = Path("fixtures/datasets/AT_008_invalid_labels.zip")
    if not dataset_zip.exists():
        raise SkipTest(f"Fixture not found: {dataset_zip}")

    files = {"file": (dataset_zip.name, dataset_zip.read_bytes(), "application/zip")}
    url = f"{ctx.api_base}/dataset/upload"
    headers = _headers(ctx).copy()
    headers.pop("Content-Type", None)
    resp = requests.post(url, headers=headers, files=files, timeout=60)

    if resp.status_code != 400:
        raise AssertionError(
            f"AT_008 expected 400, got {resp.status_code}: {resp.text}"
        )

    body = resp.json()
    msg = (body.get("error") or "") + " " + (body.get("message") or "")
    if "malformed" not in msg.lower() and "label" not in msg.lower():
        raise AssertionError(f"AT_008 expected malformed label error, got: {body}")


def test_AT_016_dataset_duplicate_version_name(ctx: TestContext) -> None:
    """
    AT_016 – Dataset validation – duplicate version name per project.

    1) Upload a valid dataset with version 'v1' for project 'TestProject'.
    2) Try another upload with same version & project.
    Expect: second call fails with conflict/validation error (409/400).
    """
    project = "TestProject_AT_016"
    version = "v1"
    dataset_zip = Path("fixtures/datasets/AT_016_valid.zip")
    if not dataset_zip.exists():
        raise SkipTest(f"Fixture not found: {dataset_zip}")

    files = {
        "file": (dataset_zip.name, dataset_zip.read_bytes(), "application/zip"),
    }
    data = {
        "project": project,
        "company": os.getenv("VISIONM_TEST_COMPANY", "RuzareInfoTech"),
        "version": version,
    }

    # First upload should succeed (201 or 200, depending on your API).
    url = f"{ctx.api_base}/dataset/upload"
    headers = _headers(ctx).copy()
    headers.pop("Content-Type", None)
    resp1 = requests.post(url, headers=headers, data=data, files=files, timeout=60)
    if resp1.status_code not in (200, 201):
        raise AssertionError(
            f"AT_016 first upload expected 200/201, got {resp1.status_code}: {resp1.text}"
        )

    # Second upload with same project+version should be rejected.
    resp2 = requests.post(url, headers=headers, data=data, files=files, timeout=60)
    if resp2.status_code not in (400, 409):
        raise AssertionError(
            f"AT_016 second upload expected 400/409, got {resp2.status_code}: {resp2.text}"
        )

    body = resp2.json()
    msg = (body.get("error") or "") + " " + (body.get("message") or "")
    if "version" not in msg.lower() and "duplicate" not in msg.lower():
        raise AssertionError(
            f"AT_016 expected duplicate version error, got: {body}"
        )


# ---------------------------------------------------------------------------
# Training / Simulation tests (subset of AT_033–AT_056)
# ---------------------------------------------------------------------------

def _start_training_job(ctx: TestContext,
                        company: str,
                        project: str,
                        dataset_id: str,
                        model_id: str,
                        epochs: int,
                        lr: float,
                        expect_ok: bool = True) -> Tuple[Optional[str], requests.Response]:
    """
    Helper to call POST /api/inference/start with a dataset + model.
    Returns (inferenceId, response).
    """
    payload = {
        "modelId": model_id,
        "datasetId": dataset_id,
        "confidenceThreshold": 0.25,
        "hyperparams": {
            "epochs": epochs,
            "learningRate": lr,
        },
        "company": company,
        "project": project,
    }
    resp = api_post(ctx, "/inference/start", payload, expected_status=202 if expect_ok else 400)
    inference_id = None
    if expect_ok:
        body = resp.json()
        inference_id = body.get("inferenceId") or body.get("id") or body.get("_id")
        if not inference_id:
            raise AssertionError(f"No inferenceId returned: {body}")
    return inference_id, resp


def test_AT_033_training_invalid_hyperparams(ctx: TestContext) -> None:
    """
    AT_033 – Training – invalid hyperparameter combination (negative epochs).

    Expect: backend validation rejects and does NOT enqueue job.
    """
    # Discover fixtures dynamically from backend
    ensure_inference_fixtures(ctx)
    company = ctx.project_ids["primary_company"]
    project = ctx.project_ids["primary_project"]
    dataset_id = ctx.dataset_ids.get("has_test_dataset")
    model_id = ctx.dataset_ids.get("primary_model")
    if not dataset_id or not model_id:
        raise SkipTest("AT_033: no dataset with tests and/or model discovered from backend")

    # Here we deliberately expect failure (400).
    try:
        _start_training_job(
            ctx,
            company,
            project,
            dataset_id,
            model_id,
            epochs=-1,
            lr=0.001,
            expect_ok=False,
        )
    except AssertionError as e:
        # If your API currently returns 200/202 even for invalid config, this will fail.
        # Once backend validation is in place, this test becomes green.
        raise AssertionError(f"AT_033 failed (hyperparam validation missing?): {e}")


def test_AT_038_training_learning_rate_too_high(ctx: TestContext) -> None:
    """
    AT_038 – Training – learning rate upper bound.

    Expect: server rejects with proper validation.
    """
    ensure_inference_fixtures(ctx)
    company = ctx.project_ids["primary_company"]
    project = ctx.project_ids["primary_project"]
    dataset_id = ctx.dataset_ids.get("has_test_dataset")
    model_id = ctx.dataset_ids.get("primary_model")
    if not dataset_id or not model_id:
        raise SkipTest("AT_038: no dataset with tests and/or model discovered from backend")

    try:
        _start_training_job(
            ctx,
            company,
            project,
            dataset_id,
            model_id,
            epochs=10,
            lr=1.0,   # way too high
            expect_ok=False,
        )
    except AssertionError as e:
        raise AssertionError(f"AT_038 failed (LR validation missing?): {e}")


# ---------------------------------------------------------------------------
# Prediction / Testing tests (subset of AT_057–AT_080)
# ---------------------------------------------------------------------------

def _start_inference_dataset_mode(ctx: TestContext,
                                  dataset_id: str,
                                  model_id: str,
                                  expect_ok: bool = True) -> Tuple[Optional[str], requests.Response]:
    payload = {
        "modelId": model_id,
        "datasetId": dataset_id,
        "confidenceThreshold": 0.25,
    }
    status = 202 if expect_ok else 400
    resp = api_post(ctx, "/inference/start", payload, expected_status=status)
    inference_id = None
    if expect_ok:
        body = resp.json()
        inference_id = body.get("inferenceId") or body.get("id") or body.get("_id")
        if not inference_id:
            raise AssertionError(f"No inferenceId returned: {body}")
    return inference_id, resp


def test_AT_057_prediction_dataset_zero_test_images(ctx: TestContext) -> None:
    """
    AT_057 – Prediction – run with dataset having zero test images.

    Expect: starting inference is rejected with clear validation error.
    """
    ensure_inference_fixtures(ctx)
    dataset_id = ctx.dataset_ids.get("zero_test_dataset")
    model_id = ctx.dataset_ids.get("primary_model")
    if not dataset_id:
        raise SkipTest("AT_057: no dataset with zero test images discovered from backend")
    if not model_id:
        raise SkipTest("AT_057: no trained model discovered from backend")

    try:
        _start_inference_dataset_mode(ctx, dataset_id, model_id, expect_ok=False)
    except AssertionError as e:
        raise AssertionError(f"AT_057 failed (zero-test-images not validated?): {e}")


def test_AT_070_prediction_confidence_threshold_impact(ctx: TestContext) -> None:
    """
    AT_070 – Prediction – verify higher threshold produces fewer detections.

    1) Run inference with threshold=0.25.
    2) Run inference with threshold=0.75.
    3) Compare totalDetections; high-threshold run should have <= low-threshold.
    """
    ensure_inference_fixtures(ctx)
    dataset_id = ctx.dataset_ids.get("has_test_dataset")
    model_id = ctx.dataset_ids.get("primary_model")
    if not dataset_id or not model_id:
        raise SkipTest("AT_070: no dataset with tests and/or model discovered from backend")

    def _start(threshold: float) -> Tuple[str, dict]:
        payload = {
            "modelId": model_id,
            "datasetId": dataset_id,
            "confidenceThreshold": threshold,
        }
        resp = api_post(ctx, "/inference/start", payload, expected_status=202)
        body = resp.json()
        infer_id = body.get("inferenceId") or body.get("id")
        if not infer_id:
            raise AssertionError(f"No inferenceId returned: {body}")

        # Poll status until completed and then fetch results
        # (simplified – you can extract to a helper)
        for _ in range(60):
            status_resp = api_get(ctx, f"/inference/{infer_id}/status")
            s_body = status_resp.json()
            if s_body.get("status") in ("completed", "failed", "cancelled"):
                break
        if s_body.get("status") != "completed":
            raise AssertionError(f"Inference {infer_id} did not complete: {s_body}")

        res_resp = api_get(ctx, f"/inference/{infer_id}/results")
        res_body = res_resp.json()
        results = res_body.get("results") or res_body
        return infer_id, results

    _, low_res = _start(0.25)
    _, high_res = _start(0.75)

    low_det = low_res.get("totalDetections", 0)
    high_det = high_res.get("totalDetections", 0)

    if not isinstance(low_det, (int, float)) or not isinstance(high_det, (int, float)):
        raise AssertionError(f"AT_070 missing totalDetections in results: {low_res}, {high_res}")

    if high_det > low_det:
        raise AssertionError(
            f"AT_070 expected high-threshold detections <= low-threshold, "
            f"got low={low_det}, high={high_det}"
        )


# ---------------------------------------------------------------------------
# Local sync / multi-company & payload chunking tests (Local app back-end)
# ---------------------------------------------------------------------------

def test_local_sync_chunking_AT_sync_payload(ctx: TestContext) -> None:
    """
    Example Local sync test (related to your 413 fixes):

    POST /api/local-sync/upload with a lot of episodes and confirm:
    - Request succeeds (no 413).
    - LocalSyncStatus is updated for (company, shopId).
    """
    company = "TestCompany"
    shop_id = "Shop1-LineA"
    episodes = [
        {"ts": f"2025-01-01T00:00:{i:02d}Z", "result": "good" if i % 2 == 0 else "defect"}
        for i in range(0, 1000)
    ]

    resp = api_post(
        ctx,
        "/local-sync/upload",
        {"company": company, "shopId": shop_id, "episodes": episodes},
        expected_status=200,
    )
    body = resp.json()
    if not body.get("ok"):
        raise AssertionError(f"Local sync upload not ok: {body}")

    # Check status endpoint
    status_resp = api_get(ctx, "/local-sync/status", params={"company": company, "shopId": shop_id})
    s_body = status_resp.json()
    if not s_body.get("lastSyncAt"):
        raise AssertionError(f"LocalSyncStatus lastSyncAt not updated: {s_body}")


# ---------------------------------------------------------------------------
# Test registry – extend to cover all 158 testcases
# ---------------------------------------------------------------------------

TEST_CASES: List[TestCase] = [
    # Dataset validation (examples)
    TestCase("AT_001", "Dataset validation – orphan labels", test_AT_001_dataset_orphan_labels),
    TestCase("AT_008", "Dataset validation – invalid label file format", test_AT_008_dataset_invalid_label_format),
    TestCase("AT_016", "Dataset validation – duplicate version name", test_AT_016_dataset_duplicate_version_name),

    # Training / Simulation
    TestCase("AT_033", "Training – invalid hyperparameter combination", test_AT_033_training_invalid_hyperparams),
    TestCase("AT_038", "Training – learning rate too high", test_AT_038_training_learning_rate_too_high),

    # Prediction / Testing
    TestCase("AT_057", "Prediction – dataset with zero test images", test_AT_057_prediction_dataset_zero_test_images),
    TestCase("AT_070", "Prediction – confidence threshold impact", test_AT_070_prediction_confidence_threshold_impact),

    # Local sync
    TestCase("LOCAL_SYNC_CHUNKING", "Local sync – chunked upload no 413 + status updated", test_local_sync_chunking_AT_sync_payload),

    # TODO:
    # - Add entries for all remaining AT_xxx IDs and map them to new test_*
    #   functions you create using the same patterns above.
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    ctx = TestContext()
    results: List[Tuple[str, bool, Optional[str]]] = []

    for tc in TEST_CASES:
        print(f"\n=== Running {tc.id}: {tc.description} ===")
        try:
            tc.fn(ctx)
            print(f"[PASS] {tc.id}")
            results.append((tc.id, True, None))
        except SkipTest as e:
            print(f"[SKIP] {tc.id}: {e}")
            results.append((tc.id, None, str(e)))
        except Exception as e:
            print(f"[FAIL] {tc.id}: {e}")
            traceback.print_exc()
            results.append((tc.id, False, str(e)))

    print("\n=== Summary ===")
    passed = sum(1 for _, ok, _ in results if ok is True)
    failed = sum(1 for _, ok, _ in results if ok is False)
    skipped = sum(1 for _, ok, _ in results if ok is None)

    for tc_id, ok, err in results:
        if ok is True:
            status = "PASS"
        elif ok is False:
            status = "FAIL"
        else:
            status = "SKIP"
        msg = "" if ok is True else f" – {err}"
        print(f"{status}: {tc_id}{msg}")

    print(f"\nTotal: {len(results)}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
