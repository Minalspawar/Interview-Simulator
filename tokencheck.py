import requests

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer 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"}

data = {
    "inputs": "Tell me a joke about data science."
}

response = requests.post(API_URL, headers=headers, json=data)
print(response.status_code)
print(response.text)
