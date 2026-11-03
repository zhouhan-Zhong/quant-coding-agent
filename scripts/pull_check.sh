#!/usr/bin/env bash
# Anonymous pullability check for the exact manifest digest named in submission.json.
#
#   scripts/pull_check.sh [registry/repository@sha256:...]
#
# Uses no local registry credentials: it requests an anonymous token from the registry and asks
# for the manifest by digest.  200 = anonymously pullable.  401/404 = NOT pullable (private
# package, wrong digest, or wrong repository) -> the organizer's pull would fail silently.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${1:-}"
if [ -z "$REF" ]; then
  REF="$(python3 -c "import json;d=json.load(open('$HERE/submission.json'))['image'];print(d['registry']+'/'+d['repository']+'@'+d['digest'])")"
fi
REGISTRY="${REF%%/*}"; REST="${REF#*/}"; REPO="${REST%%@*}"; DIGEST="${REST#*@}"
ACCEPT='application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json'

case "$REGISTRY" in
  docker.io|index.docker.io)
    HOST="registry-1.docker.io"
    TOKEN="$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("token",""))')" ;;
  ghcr.io)
    HOST="ghcr.io"
    TOKEN="$(curl -s "https://ghcr.io/token?scope=repository:$REPO:pull&service=ghcr.io" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("token",""))')" ;;
  *)
    HOST="$REGISTRY"; TOKEN="" ;;
esac

CODE="$(curl -s -o /tmp/manifest.json -w '%{http_code}' -H "Authorization: Bearer $TOKEN" -H "Accept: $ACCEPT" "https://$HOST/v2/$REPO/manifests/$DIGEST")"
echo "GET https://$HOST/v2/$REPO/manifests/$DIGEST -> HTTP $CODE"
if [ "$CODE" = "200" ]; then
  echo "OK: anonymously pullable."
  python3 - <<'EOF'
import json
m = json.load(open("/tmp/manifest.json"))
mt = m.get("mediaType", "?")
print("mediaType:", mt)
if "manifests" in m:
    print("this digest is an image INDEX with", len(m["manifests"]), "entries:")
    for e in m["manifests"]:
        print("   ", e.get("platform"), e.get("digest"))
    print("note: an index is pullable by digest too; single-arch linux/amd64 is what the fleet needs.")
EOF
else
  echo "NOT pullable anonymously. For GHCR: open the package on github.com -> Package settings -> Change visibility -> Public."
  exit 1
fi
