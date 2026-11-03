#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build submission.zip (submission.json + team-claim.json) with the standard library only.

Byte-for-byte the same archive `qfbench2 submission pack` writes (verified against toolkit v2.5.1),
so it can run on a plain Windows/macOS/Linux Python >= 3.8 without the toolkit, git or Docker.

    python pack_standalone.py --team-number 42                       # asks for the Team Key (hidden)
    python pack_standalone.py --team-number 42 --team-key-file key.txt
    python pack_standalone.py --team-number 42 --alias               # only print the derived team_id

Input : submission.json (the descriptor; `team_id` and `descriptor_digest` may be absent - they are
        derived and sealed here).  Output: submission.zip, ready to upload on CodaBench.

The Team Key is read from a hidden prompt or a file you own; it is never printed, never written
anywhere and never put into the zip.  The zip carries only an HMAC proof bound to this descriptor.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import math
import os
import re
import stat
import sys
import zipfile
from pathlib import Path

ALIAS_PREFIX = b"agenthon2026-team-alias:"
PROOF_KEY_PREFIX = b"agenthon2026-team-proof-key:v2:"
CLAIM_MESSAGE_PREFIX = b"agenthon2026-team-claim:v2:"
CLAIM_SCHEMA_VERSION = "2.0"
TOP_KEYS = ("schema_version", "interface_version", "competition_id", "team_id", "track", "phase",
            "category", "image", "image_access", "models", "license", "descriptor_digest")
MODEL_KEYS = ("name", "version", "training_cutoff", "access", "revision")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


# ----------------------------------------------------------------------------- RFC 8785 (JCS)
_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _jcs_string(value: str) -> str:
    out = ['"']
    for ch in value:
        esc = _ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ch < "\x20":
            out.append(f"\\u{ord(ch):04x}")
        elif "\ud800" <= ch <= "\udfff":
            raise ValueError("lone surrogate in a string")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _es_number(x: float) -> str:
    """ECMAScript Number::toString for a finite double (RFC 8785 section 3.2.2.3)."""
    if math.isnan(x) or math.isinf(x):
        raise ValueError("NaN/Infinity cannot be canonicalized")
    if x == 0:
        return "0"
    if x < 0:
        return "-" + _es_number(-x)
    r = repr(x)
    if "e" in r or "E" in r:
        mantissa, _, exponent = r.replace("E", "e").partition("e")
        exp = int(exponent)
    else:
        mantissa, exp = r, 0
    int_part, _, frac_part = mantissa.partition(".")
    raw = int_part + frac_part
    stripped = raw.lstrip("0")
    n = len(int_part) + exp - (len(raw) - len(stripped))
    digits = stripped.rstrip("0") or "0"
    k = len(digits)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    head = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{head}e{'+' if e >= 0 else '-'}{abs(e)}"


def _jcs(value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_string(value)
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError("integer outside the exact double range")
        return str(value)
    if isinstance(value, float):
        return _es_number(value)
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]).encode("utf-16-be"))
        return "{" + ",".join(_jcs_string(k) + ":" + _jcs(v) for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_jcs(v) for v in value) + "]"
    raise ValueError(f"{type(value).__name__} has no JSON form")


def jcs_canonical(obj) -> bytes:
    return _jcs(obj).encode("utf-8")


def seal_descriptor_digest(descriptor: dict) -> dict:
    body = {k: v for k, v in descriptor.items() if k != "descriptor_digest"}
    digest = "sha256:" + hashlib.sha256(jcs_canonical(body)).hexdigest()
    sealed = dict(body)
    sealed["descriptor_digest"] = digest
    return sealed


# ----------------------------------------------------------------------------- team identity
def validate_team_key(key: str) -> str:
    if not isinstance(key, str) or not 8 <= len(key) <= 256:
        raise SystemExit("refused: team key must be 8..256 characters")
    if not key.isprintable() or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in key):
        raise SystemExit("refused: team key must be printable, with no control characters")
    return key


def derive_team_alias(team_number: int, team_key: str) -> str:
    material = ALIAS_PREFIX + str(team_number).encode("ascii") + b":" + team_key.encode("utf-8")
    return "team-" + hashlib.sha256(material).hexdigest()[:32]


def team_claim_proof(team_number: int, team_key: str, descriptor_sha256: str) -> str:
    proof_key = hashlib.sha256(PROOF_KEY_PREFIX + team_key.encode("utf-8")).digest()
    message = CLAIM_MESSAGE_PREFIX + str(team_number).encode("ascii") + b":" + descriptor_sha256.encode("ascii")
    return hmac.new(proof_key, message, hashlib.sha256).hexdigest()


def build_team_claim(team_number: int, team_key: str, descriptor_sha256: str) -> bytes:
    claim = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "site_team_id": team_number,
        "descriptor_sha256": descriptor_sha256,
        "proof": team_claim_proof(team_number, team_key, descriptor_sha256),
    }
    return json.dumps(claim, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


# ----------------------------------------------------------------------------- descriptor checks
def check_descriptor(desc: dict, phase_hint: str | None) -> None:
    unknown = sorted(set(desc) - set(TOP_KEYS))
    if unknown:
        raise SystemExit(f"refused: unknown descriptor field(s) {unknown}; the vocabulary is closed")
    missing = [k for k in TOP_KEYS if k not in desc and k not in ("team_id", "descriptor_digest")]
    if missing:
        raise SystemExit(f"refused: descriptor is missing {missing}")
    if desc["interface_version"] != "2.0":
        raise SystemExit("refused: interface_version must be \"2.0\"")
    if desc["schema_version"] not in ("1.0.0", "1.1.0"):
        raise SystemExit("refused: schema_version must be 1.0.0 or 1.1.0")
    if desc["category"] not in ("api", "simulator"):
        raise SystemExit("refused: category must be api (Track 1)")
    if desc["track"] != "coding":
        raise SystemExit("refused: track must be coding for this kit")
    if desc["phase"] not in ("dev", "final", "verification"):
        raise SystemExit("refused: phase must be dev|final|verification")
    if desc["competition_id"] != f"agenthon2026-coding-{desc['phase']}":
        raise SystemExit(f"refused: competition_id must be agenthon2026-coding-{desc['phase']} for phase {desc['phase']}")
    img = desc["image"]
    if not isinstance(img, dict) or set(img) != {"registry", "repository", "digest"}:
        raise SystemExit("refused: image must be an object with exactly registry/repository/digest")
    if not DIGEST_RE.match(str(img["digest"])):
        raise SystemExit("refused: image.digest must be sha256:<64 lowercase hex>")
    fixture_digests = {"sha256:9586b834b7ab2ce09dbb93d4cc00fff8da6a408f226beb1e1ffb8870165de488",
                       "sha256:a17eb95d89902977aaf45034f7749935a7b4b03755b162116dd726a39c3af0ae"}
    if "PLACEHOLDER" in json.dumps(img) or img["repository"] == "team-example/qfb2-solver" or img["digest"] in fixture_digests:
        raise SystemExit("refused: image still carries the organizers' example value; push your own image and use its digest")
    if desc["image_access"] not in ("public", "organizer_mirror"):
        raise SystemExit("refused: image_access must be public or organizer_mirror")
    if not isinstance(desc["models"], list):
        raise SystemExit("refused: models must be an array")
    for i, m in enumerate(desc["models"]):
        if not isinstance(m, dict) or set(m) != set(MODEL_KEYS):
            raise SystemExit(f"refused: models[{i}] must have exactly the keys {MODEL_KEYS}")
        if m["access"] not in ("api", "local"):
            raise SystemExit(f"refused: models[{i}].access must be api or local")
    if not desc["models"]:
        print("warning: models is empty - on Track 1 that validates but earns no credit", file=sys.stderr)
    if phase_hint and phase_hint != desc["phase"]:
        raise SystemExit(f"refused: descriptor phase is {desc['phase']} but --phase {phase_hint} was requested")


# ----------------------------------------------------------------------------- packing
def read_team_key(args) -> str:
    if args.team_key_file:
        p = Path(args.team_key_file)
        raw = p.read_bytes()
        if len(raw) > 4096:
            raise SystemExit("refused: key file is too large to be a key")
        # Windows editors may add a byte-order mark; PowerShell 5 ">" writes UTF-16.  The key itself is unchanged.
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8-sig")
        if text.endswith("\r\n"):
            text = text[:-2]
        elif text.endswith("\n"):
            text = text[:-1]
        return validate_team_key(text)
    if not sys.stdin.isatty():
        raise SystemExit("refused: no terminal to hide the Team Key prompt; use --team-key-file")
    key = getpass.getpass("Team Key (hidden, not echoed): ")
    return validate_team_key(key)


def write_private(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise SystemExit("refused: output path exists and is not a regular file")
        path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(payload)


def pack(descriptor: dict, team_number: int, team_key: str, out: Path) -> tuple[str, str]:
    alias = derive_team_alias(team_number, team_key)
    declared = descriptor.get("team_id")
    if declared is not None and declared != alias:
        raise SystemExit("refused: descriptor team_id disagrees with the id derived from this team number and key; "
                         "remove team_id from submission.json or check the number and key")
    body = {k: v for k, v in descriptor.items() if k != "descriptor_digest"}
    body["team_id"] = alias
    sealed = seal_descriptor_digest(body)
    descriptor_bytes = (json.dumps(sealed, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor_sha256 = hashlib.sha256(descriptor_bytes).hexdigest()
    claim = build_team_claim(team_number, team_key, descriptor_sha256)

    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in (("submission.json", descriptor_bytes), ("team-claim.json", claim)):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            entry.compress_type = zipfile.ZIP_STORED
            archive.writestr(entry, payload)
    write_private(out, buf.getvalue())
    return alias, descriptor_sha256


def verify_zip(out: Path, team_number: int, team_key: str) -> None:
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        if names != ["submission.json", "team-claim.json"]:
            raise SystemExit(f"self-check failed: zip members are {names}")
        desc_bytes = archive.read("submission.json")
        claim = json.loads(archive.read("team-claim.json"))
    digest = hashlib.sha256(desc_bytes).hexdigest()
    if claim["descriptor_sha256"] != digest:
        raise SystemExit("self-check failed: claim is not bound to the descriptor bytes")
    if claim["proof"] != team_claim_proof(team_number, team_key, digest):
        raise SystemExit("self-check failed: proof does not verify")
    desc = json.loads(desc_bytes)
    resealed = seal_descriptor_digest(desc)
    if resealed["descriptor_digest"] != desc["descriptor_digest"]:
        raise SystemExit("self-check failed: descriptor_digest does not reseal to itself")


def main() -> int:
    ap = argparse.ArgumentParser(description="pack submission.zip for Agenthon 2026 (standard library only)")
    ap.add_argument("--team-number", type=int, required=True, help="your team number from agenthon.net")
    ap.add_argument("--descriptor", default="submission.json")
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--team-key-file", help="read the Team Key from this file (mode 600) instead of the hidden prompt")
    ap.add_argument("--phase", choices=["dev", "final", "verification"], help="refuse unless the descriptor is for this phase")
    ap.add_argument("--alias", action="store_true", help="only print the derived team_id and exit")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output zip")
    args = ap.parse_args()
    if args.team_number <= 0:
        raise SystemExit("refused: team number must be a positive integer")
    for a in sys.argv[1:]:
        if a.startswith("--team-key") and not a.startswith("--team-key-file"):
            raise SystemExit("refused: the Team Key is never a command-line argument; use the hidden prompt or --team-key-file")

    team_key = read_team_key(args)
    if args.alias:
        print(derive_team_alias(args.team_number, team_key))
        return 0

    desc_path = Path(args.descriptor)
    try:
        descriptor = json.loads(desc_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"refused: {desc_path} is not a readable JSON file ({exc})")
    if not isinstance(descriptor, dict):
        raise SystemExit("refused: descriptor must be a JSON object")
    check_descriptor(descriptor, args.phase)

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"refused: {out} exists; pass --force to replace it")
    alias, digest = pack(descriptor, args.team_number, team_key, out)
    verify_zip(out, args.team_number, team_key)
    print(f"wrote {out} (submission.json + team-claim.json)")
    print(f"team_id          : {alias}")
    print(f"descriptor_sha256: {digest}")
    print(f"image            : {descriptor['image']['registry']}/{descriptor['image']['repository']}@{descriptor['image']['digest']}")
    print(f"phase            : {descriptor['phase']} ({descriptor['competition_id']})")
    print("self-check       : claim verifies against the zip's own submission.json bytes")
    print(f"Upload this zip as it is on the CodaBench Track 1 page -> My Submissions ({descriptor['phase']} phase).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
