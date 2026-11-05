"""Shared finite JSON and fingerprints, with no solver or renderer imports."""
import hashlib,json
from pathlib import Path
from assembly.paths import input_path,output_path


def read(path):return json.loads(input_path(path).read_text())


def dump(path,data):
    path=output_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
