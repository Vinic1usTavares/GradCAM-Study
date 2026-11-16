import os
import hashlib
import json

MANIFEST = {
    "cat_dog": "notebooks/assets/cat_dog.jpg"
}

def get_manifest_hash():
    hasher = hashlib.sha1()
    for k, path in sorted(MANIFEST.items()):
        if os.path.exists(path):
            with open(path, "rb") as f:
                hasher.update(f.read())
    return hasher.hexdigest()
