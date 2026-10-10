"""Versioned, checksummed artifacts and atomic journals (not an OS access sandbox)."""

import hashlib
import json
import os
from pathlib import Path


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse(text):
    return json.loads(
        text,
        object_pairs_hook=unique_fields,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def seal(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    atomic_json(path, {"sha256": fingerprint(payload), "payload": payload})
    return payload


def unseal(path):
    value = parse(Path(path).read_text())
    if set(value) != {"sha256", "payload"} or value["sha256"] != fingerprint(value["payload"]):
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return value["payload"]


def read_json(path):
    return parse(Path(path).read_text())


class RunLock:
    """Prevent concurrent mutation and duplicate paid calls to one experiment."""

    def __init__(self, root):
        self.path = Path(root) / ".writer.lock"

    def __enter__(self):
        try:
            with self.path.open("x") as stream:
                stream.write(str(os.getpid()))
        except FileExistsError:
            raise ValueError(
                "Workspace writer lock exists; check its process before removing it"
            ) from None
        return self

    def __exit__(self, *args):
        self.path.unlink()
