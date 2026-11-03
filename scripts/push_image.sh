#!/usr/bin/env bash
# Tag + push the agent image to a public registry and write submission.json with its digest.
#
#   scripts/push_image.sh <registry/repository:tag> [local-image-tag]
#   e.g. scripts/push_image.sh docker.io/yourname/agenthon-t1:v1
#        scripts/push_image.sh ghcr.io/yourname/agenthon-t1:v1
#
# You must be logged in first (docker login / docker login ghcr.io).  The image must end up
# ANONYMOUSLY pullable: Docker Hub repos are public by default; GHCR packages are PRIVATE by
# default and must be switched to public in the package settings on github.com.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${1:?registry/repository:tag}"
LOCAL="${2:-agenthon-t1-agent:dev}"

case "$REF" in
  */*) ;;
  *) echo "give a full reference like docker.io/<user>/<repo>:<tag>" >&2; exit 2 ;;
esac
# Docker Hub short form user/repo:tag -> docker.io/user/repo:tag
FIRST="${REF%%/*}"
if [[ "$FIRST" != *.* && "$FIRST" != *:* && "$FIRST" != localhost ]]; then REF="docker.io/$REF"; fi
case "$REF" in *:*) ;; *) echo "add an explicit :tag to the reference" >&2; exit 2 ;; esac

docker tag "$LOCAL" "$REF"
echo "== pushing $REF"
PUSH_LOG="$(mktemp)"
docker push "$REF" 2>&1 | tee "$PUSH_LOG"

REPO_NO_TAG="${REF%:*}"            # registry/repository
REGISTRY="${REPO_NO_TAG%%/*}"
REPOSITORY="${REPO_NO_TAG#*/}"

# 1) the digest line printed by `docker push` is the pushed manifest digest
DIGEST="$(grep -oE 'digest: sha256:[0-9a-f]{64}' "$PUSH_LOG" | tail -1 | sed 's/digest: //')"
# 2) fallback: RepoDigests of the local image (docker.io entries carry no registry prefix)
if [ -z "$DIGEST" ]; then
  DIGEST="$(docker inspect --format '{{json .RepoDigests}}' "$REF" | python3 -c '
import json, sys
repo = sys.argv[1]; short = repo.split("/", 1)[1]
for d in json.load(sys.stdin):
    name, _, dig = d.partition("@")
    if name in (repo, short, "docker.io/" + short):
        print(dig); break
' "$REPO_NO_TAG")"
fi
# 3) fallback: ask the registry
if [ -z "$DIGEST" ]; then
  DIGEST="$(docker buildx imagetools inspect "$REF" --format '{{.Manifest.Digest}}' 2>/dev/null || true)"
fi
rm -f "$PUSH_LOG"
if [ -z "$DIGEST" ]; then echo "could not determine the pushed digest; run: docker buildx imagetools inspect $REF" >&2; exit 1; fi

echo "pushed: $REPO_NO_TAG@$DIGEST"
python3 "$HERE/scripts/make_descriptor.py" --registry "$REGISTRY" --repository "$REPOSITORY" --digest "$DIGEST" --out "$HERE/submission.json"
echo
echo "Next: scripts/pull_check.sh          (anonymous pull check of $REPO_NO_TAG@$DIGEST)"
echo "Then: scripts/pack.sh <team-number>  (writes submission.zip)"
