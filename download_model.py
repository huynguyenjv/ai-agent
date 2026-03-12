"""
Script để download embedding model và export sang ONNX format.

Chạy script này trên máy có PyTorch + sentence-transformers, sau đó chỉ cần
copy folder models/all-MiniLM-L6-v2-onnx/ sang máy triển khai.

Máy triển khai chỉ cần onnxruntime + tokenizers (không cần PyTorch).

Usage:
    pip install sentence-transformers torch onnxscript
    python download_model.py
"""

import os
import json
import ssl
import warnings

# Bypass SSL for corporate environments
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
except Exception:
    pass

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_PATH = "./models/all-MiniLM-L6-v2-onnx"


def main():
    print(f"Downloading model: {MODEL_NAME}")
    print(f"ONNX output:       {ONNX_PATH}")
    print("-" * 50)

    # 1. Load model via SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    transformer = model[0]
    tokenizer = transformer.tokenizer

    # 2. Export to ONNX
    os.makedirs(ONNX_PATH, exist_ok=True)
    print("Exporting to ONNX ...")

    dummy = tokenizer("test input", return_tensors="pt", padding=True)
    torch.onnx.export(
        transformer.auto_model,
        (dummy["input_ids"], dummy["attention_mask"],
         dummy.get("token_type_ids", dummy["attention_mask"])),
        f"{ONNX_PATH}/model.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # 3. Save tokenizer
    tokenizer.save_pretrained(ONNX_PATH)

    # 4. Save pooling config
    pooling_config = model[1].get_config_dict()
    with open(f"{ONNX_PATH}/pooling_config.json", "w") as f:
        json.dump(pooling_config, f, indent=2)

    # 5. Save embedding config
    emb_dim = model.get_sentence_embedding_dimension()
    with open(f"{ONNX_PATH}/embedding_config.json", "w") as f:
        json.dump({
            "embedding_dimension": emb_dim,
            "max_seq_length": model.max_seq_length,
            "normalize": True,
        }, f, indent=2)

    print("-" * 50)
    onnx_size = os.path.getsize(f"{ONNX_PATH}/model.onnx") / 1024 / 1024
    data_file = f"{ONNX_PATH}/model.onnx.data"
    data_size = os.path.getsize(data_file) / 1024 / 1024 if os.path.exists(data_file) else 0
    print(f"✓ Model exported: {onnx_size:.1f} MB (+{data_size:.1f} MB weights)")
    print(f"✓ Embedding dim:  {emb_dim}")
    print(f"✓ Output dir:     {ONNX_PATH}")
    print()
    print("Copy models/all-MiniLM-L6-v2-onnx/ to target machine.")
    print("Target machine only needs: onnxruntime, tokenizers, numpy")
    print()
    print("Để sử dụng model offline, update .env:")
    print(f"  EMBEDDING_MODEL=all-MiniLM-L6-v2-onnx")

if __name__ == "__main__":
    main()

