import sys, pathlib, datetime as dt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src.config_loader import load_env
load_env()
from src import alert, meta_live

res = meta_live.predict_meta(filename="xau_usd_4h_real.csv", horizon=4, threshold=0.50)
trend = res.get("trend", 0)
p_up = res.get("probability", 0.0)

emoji_map = {1: "\U0001f4c8", -1: "\U0001f4c9", 0: "\U0001f6cc"}  # 📈 📉 🛌
conf = "\U0001f4aa STRONG" if p_up >= 0.60 else "\U0001f44d MEDIUM" if p_up >= 0.55 else "\U0001f7e1 WEAK"
trend_label = {1: "TRENDING UP", -1: "TRENDING DOWN", 0: "NEUTRAL / CHOPPY"}[trend]
side = res.get("side") or {1: "BUY/SELL", 0: "WAIT"}[trend]
status_line = "\u2705 TRADE" if res.get("trade") else "\u23f8\ufe0f NO TRADE"

msg = (
    f"{emoji_map[trend]} Cerebrum Meta-Status\n"
    f"Trend: {trend_label}  |  Confidence: {conf} ({p_up:.1%})\n"
    f"Side: {side}  |  Status: {status_line}\n"
    f"Reason: {res['reason']}\n"
    f"Engine: meta-labeling (trained {res['trained_rows']} bars) | Next check: +4h"
)

sent = alert.send_message(msg)
print("SENT" if sent else "NOT SENT")
