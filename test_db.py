import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from src.db_utils import init_db_pool, create_tables, log_prediction, fetch_recent_predictions

init_db_pool()
create_tables()

rid = log_prediction(
    input_text="Tesla reports record quarterly revenue.",
    predicted_label="Business",
    confidence=0.936,
    all_scores=[
        {"label": "Business", "score": 0.936},
        {"label": "Science/Technology", "score": 0.057},
    ],
)
print(f"Logged with ID: {rid}")

records = fetch_recent_predictions(limit=3)
for r in records:
    print(r)