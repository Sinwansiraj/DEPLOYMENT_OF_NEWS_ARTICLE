# test_inference_quick.py
import sys
sys.path.insert(0, ".")
from src.inference import load_model, predict

load_model("model_artifacts/final_model")

tests = [
    "Apple launches new iPhone with breakthrough AI chip",
    "Manchester United beats Arsenal in extra time thriller",
    "Federal Reserve raises interest rates by 50 basis points",
    "NASA discovers evidence of water ice beneath Mars surface",
]

for t in tests:
    r = predict(t)
    print(f"[{r['predicted_label']:20s}] {r['confidence']*100:.1f}%  →  {t}")