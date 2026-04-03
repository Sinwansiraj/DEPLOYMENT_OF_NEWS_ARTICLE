#!/usr/bin/env bash
# =============================================================================
# deploy.sh — EC2 deployment script for News Categorization System
# Run this once on a fresh Ubuntu 22.04 EC2 instance (t3.medium or better).
# =============================================================================

set -euo pipefail
APP_DIR="/home/ubuntu/news-categorization"
PORT=8501

echo "=============================="
echo "  News Categorizer Deployment"
echo "=============================="

# ── 1. System packages ─────────────────────────────────────────────────────
echo "[1/7] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    git curl unzip \
    libpq-dev gcc     # required for psycopg2

# ── 2. Clone / update repo ─────────────────────────────────────────────────
echo "[2/7] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull
else
    git clone https://github.com/Sinwansiraj/DEPLOYMENT_OF_NEWS_ARTICLE.git "$APP_DIR"
    cd "$APP_DIR"
fi

# ── 3. Python virtual environment ──────────────────────────────────────────
echo "[3/7] Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# ── 4. Environment variables ───────────────────────────────────────────────
echo "[4/7] Configuring environment variables..."
# Replace these with your actual values or use AWS SSM Parameter Store
cat > "$APP_DIR/.env" <<EOF
AWS_REGION=us-east-1
S3_BUCKET=your-news-categorizer-bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
DB_HOST=your-rds-endpoint.amazonaws.com
DB_PORT=5432
DB_NAME=news_categorizer
DB_USER=postgres
DB_PASSWORD=your_db_password
EOF

export $(grep -v '^#' "$APP_DIR/.env" | xargs)

# ── 5. Download model from S3 ──────────────────────────────────────────────
echo "[5/7] Pulling model from S3..."
python3 - <<'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from src.aws_utils import download_model_from_s3
from src.inference import load_model
import tarfile

archive = "model_artifacts/model.tar.gz"
os.makedirs("model_artifacts", exist_ok=True)
if not os.path.exists("model_artifacts/final_model"):
    download_model_from_s3(archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall("model_artifacts")
    print("Model extracted.")
else:
    print("Model already present, skipping download.")
PYEOF

# ── 6. Initialize database tables ─────────────────────────────────────────
echo "[6/7] Initializing database..."
python3 - <<'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from src.db_utils import init_db_pool, create_tables
init_db_pool()
create_tables()
print("Database tables ready.")
PYEOF

# ── 7. Start Streamlit ─────────────────────────────────────────────────────
echo "[7/7] Starting Streamlit on port $PORT..."
nohup venv/bin/streamlit run app.py \
    --server.port "$PORT" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    > /home/ubuntu/streamlit.log 2>&1 &

echo ""
echo "✅ Deployment complete!"
echo "   App running at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):${PORT}"
echo "   Logs: tail -f /home/ubuntu/streamlit.log"