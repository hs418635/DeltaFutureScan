import requests

BOT_TOKEN = "8809200223:AAHmR969mpLw_jEuH2iVeLJ1RcGY8DeosXA"  # Replace with your actual token
CHAT_ID = "503404993"      # Replace with your actual Chat ID

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {
    'chat_id': CHAT_ID,
    'text': "🤖 Test message from Delta Scanner Bot!"
}

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
#