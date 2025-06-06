# scripts/hash_utils.py

import hashlib

def compute_md5(file_path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return hashlib.md5(file_bytes).hexdigest()
