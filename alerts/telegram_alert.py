import requests

BOT_TOKEN = "PASTE_YOUR_BOT_TOKEN"
CHAT_ID = "PASTE_YOUR_CHAT_ID"

def send_telegram_alert(message="🚨 Violence detected!"):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    requests.post(url, data=data)
