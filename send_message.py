import requests


def send_message(token, user_id, message):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = {
        "to": user_id,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    response = requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=data)
    return response


