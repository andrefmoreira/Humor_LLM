import requests

base_url = "http://crai-04.dei.uc.pt:8080"

response = requests.get(f"{base_url}/api/tags")

if response.status_code == 200:
    models = response.json().get("models", [])
    print("Available models:")
    for model in models:
        print(f"- {model['name']}")
else:
    print("Failed to fetch models:", response.text)


'''
Available models:
- gemma3:27b
- deepseek-r1:70b
- llama3.3:latest
- llava:34b-v1.6
- nomic-embed-text:latest
- gemma:text
'''