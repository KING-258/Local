import random
from datetime import datetime, timedelta, time
import json

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime.now().date()
MIN_RECORDS_PER_DAY = 180
MAX_RECORDS_PER_DAY = 230
GOOD_PROBABILITY = 0.92
OUTPUT_FILE = "local_stats.json"

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
        ts = datetime.combine(current_date, random_time())
        result = "good" if random.random() < GOOD_PROBABILITY else "defect"

        data.append({
            "ts": ts.isoformat(timespec="seconds"),
            "result": result
        })

    current_date += timedelta(days=1)
data.sort(key=lambda x: x["ts"])
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
