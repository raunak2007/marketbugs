import hashlib, json


def sha(obj) -> str:
    if not isinstance(obj, (str, bytes)):
        obj = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    if isinstance(obj, str):
        obj = obj.encode()
    return hashlib.sha256(obj).hexdigest()[:16]
