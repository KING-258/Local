#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [ ! -d ".env" ]; then
    echo "Error: .env virtual environment not found in $(pwd)"
    exit 1
fi

source .env/bin/activate

LOCAL_API_BASE_URL="http://192.168.1.24:3000/api" \
LOCAL_COMPANY_NAME="RuzareInfoTech" \
LOCAL_SHOP_ID="Shop1-LineA" \
python local_app.py