#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write submission.json (the C5 descriptor) for the Track 1 Development phase.

    python scripts/make_descriptor.py --image docker.io/<user>/<repo>@sha256:<64hex> [--out submission.json]
    python scripts/make_descriptor.py --registry ghcr.io --repository <user>/<repo> --digest sha256:<64hex>

`team_id` and `descriptor_digest` are deliberately left out: `qfbench2 submission pack` derives the
team id from your team number + Team Key, seals the digest and writes the zip.  If the toolkit is
installed the descriptor is validated here as well (with a placeholder team id).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HOUSE_MODEL = {
    "name": "nvidia/nemotron-3-super-120b-a12b",
    "version": "rl-030326-fp8",
    "revision": "rl-030326-fp8",
    "training_cutoff": "unpublished",
    "access": "api",
}

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="full reference registry/repository@sha256:<digest>")
    ap.add_argument("--registry")
    ap.add_argument("--repository")
    ap.add_argument("--digest")
    ap.add_argument("--phase", default="dev", choices=["dev", "final", "verification"])
    ap.add_argument("--license", default="Apache-2.0")
    ap.add_argument("--out", default="submission.json")
    args = ap.parse_args()

    registry, repository, digest = args.registry, args.repository, args.digest
    if args.image:
        m = re.match(r"^([^/]+)/(.+)@(sha256:[0-9a-f]{64})$", args.image.strip())
        if not m:
            print("--image must look like registry/repository@sha256:<64 hex>", file=sys.stderr)
            return 2
        registry, repository, digest = m.group(1), m.group(2), m.group(3)
    if not (registry and repository and digest):
        print("give --image or all of --registry/--repository/--digest", file=sys.stderr)
        return 2
    if not DIGEST_RE.match(digest):
        print("digest must be sha256:<64 lowercase hex> (an immutable manifest digest, not a tag)", file=sys.stderr)
        return 2
    if registry in ("docker.io", "index.docker.io") and "/" not in repository:
        repository = "library/" + repository

    desc = {
        "schema_version": "1.0.0",  # the value in the organizers' published fixture (1.1.0 is also accepted)
        "interface_version": "2.0",
        "competition_id": f"agenthon2026-coding-{args.phase}",
        "track": "coding",
        "phase": args.phase,
        "category": "api",
        "image": {"registry": registry, "repository": repository, "digest": digest},
        "image_access": "public",
        "models": [HOUSE_MODEL],
        "license": args.license,
    }
    # Validate with the toolkit when available (placeholder team id; pack replaces it).
    try:
        from qfbench2_common.contracts.descriptor import SubmissionDescriptor, seal_descriptor_digest

        probe = dict(desc)
        probe["team_id"] = "team-" + "0" * 32
        SubmissionDescriptor.from_mapping(seal_descriptor_digest(probe))
        print("descriptor validates against the installed toolkit (C5 1.1.0)")
    except ImportError:
        print("note: qfbench2-common is not installed here; validation skipped (pack will validate)")
    except Exception as exc:  # pragma: no cover
        print(f"descriptor INVALID: {exc}", file=sys.stderr)
        return 1
    Path(args.out).write_text(json.dumps(desc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: image {registry}/{repository}@{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
