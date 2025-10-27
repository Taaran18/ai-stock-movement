import os, json, requests, subprocess

TOKEN = os.getenv("TG_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")


def predict(ticker="AAPL"):
    out = subprocess.check_output(
        ["python", "predict.py", "--ticker", ticker], text=True
    )
    return json.loads(out)


if __name__ == "__main__":
    res = predict("AAPL")
    msg = f"📈 {res['ticker']} {res['date']}\nPrediction: *{res['label']}*\nProba: {res['proba']}"
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
    )
