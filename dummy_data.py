import random
from datetime import datetime, timedelta, time, timezone
import json
import sqlite3
from pathlib import Path

# Use explicit IST timezone (UTC+5:30) so all dummy timestamps are
# generated according to India Standard Time, independent of OS TZ.
IST = timezone(timedelta(hours=5, minutes=30))

START_DATE = datetime(2026, 1, 1, tzinfo=IST)
END_DATE = (datetime.now(IST) - timedelta(days=1)).date()
MIN_RECORDS_PER_DAY = 180
MAX_RECORDS_PER_DAY = 230
GOOD_PROBABILITY = 0.92
OUTPUT_FILE = "local_stats.json"
DB_FILE = "local_stats.db"

def random_time(start_hour=8, end_hour=18):
    hour = random.randint(start_hour, end_hour - 1)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return time(hour, minute, second)

data = []
current_date = START_DATE.date()

while current_date <= END_DATE:
    records_today = random.randint(
        MIN_RECORDS_PER_DAY,
        MAX_RECORDS_PER_DAY
    )

    for _ in range(records_today):
        # Combine date with random time in IST, then convert to ISO string.
        ts = datetime.combine(current_date, random_time(), tzinfo=IST)
        result = "good" if random.random() < GOOD_PROBABILITY else "defect"

        data.append({
            "ts": ts.isoformat(timespec="seconds"),
            "result": result
        })

    current_date += timedelta(days=1)
data.sort(key=lambda x: x["ts"])
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

# Keep SQLite database in sync with the JSON dummy data so the
# Local app sees the same episodes whether it reads from JSON or DB.

db_path = Path(DB_FILE)
with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()

    # Schema mirrors MainWindow._init_db in local_app.py
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            result TEXT NOT NULL CHECK (result IN ('good', 'defect'))
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_synced_ts TEXT,
            last_sync_at TEXT
        )
        """
    )
    cur.execute(
        "INSERT OR IGNORE INTO sync_state (id, last_synced_ts, last_sync_at) VALUES (1, NULL, NULL)"
    )

    # Replace existing episodes with the freshly generated dummy data.
    cur.execute("DELETE FROM episodes")
    payload = [(e["ts"], e["result"]) for e in data]
    cur.executemany("INSERT INTO episodes (ts, result) VALUES (?, ?)", payload)

    conn.commit()
