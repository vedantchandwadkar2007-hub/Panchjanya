"""
Quick check: lists every model your Groq API key currently has access to.
Run this from your hackathon folder (it reads GROQ_API_KEY from .env).
"""
import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY not found in .env — check your .env file.")
else:
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
    else:
        models = resp.json().get("data", [])
        print(f"\n{len(models)} models available to your key:\n")
        for m in sorted(models, key=lambda x: x["id"]):
            print(f"  {m['id']}")
