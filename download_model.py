"""
Script để download embedding model offline.
Chạy script này trên máy có internet tốt, sau đó copy folder models/ sang máy cần dùng.

Usage:
    python download_model.py
"""

import os
import ssl
import warnings

# Bypass SSL
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

# Patch requests
try:
    import requests
    old_request = requests.Session.request
    def new_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return old_request(self, *args, **kwargs)
    requests.Session.request = new_request
except:
    pass

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_PATH = "./models/all-MiniLM-L6-v2"

def main():
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Saving to: {LOCAL_PATH}")
    print("-" * 50)
    
    # Download và save model
    model = SentenceTransformer(MODEL_NAME)
    
    # Save to local path
    os.makedirs(LOCAL_PATH, exist_ok=True)
    model.save(LOCAL_PATH)
    
    print("-" * 50)
    print(f"✓ Model saved to: {LOCAL_PATH}")
    print()
    print("Để sử dụng model offline, update .env:")
    print(f"  EMBEDDING_MODEL={LOCAL_PATH}")

if __name__ == "__main__":
    main()

