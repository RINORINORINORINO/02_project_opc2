import requests
import os

# .env 파일에서 API 키 로드
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")

url = "https://api.elevenlabs.io/v1/voices"
headers = {
    "Accept": "application/json",
    "xi-api-key": api_key
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    voices = response.json().get("voices", [])
    print(f"총 {len(voices)}개 음성 찾음:")
    for voice in voices:
        print(f"이름: {voice.get('name')}, ID: {voice.get('voice_id')}")
else:
    print(f"API 요청 실패: {response.status_code}")