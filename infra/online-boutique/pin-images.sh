#!/usr/bin/env bash
#
# pin-images.sh — resolve every image referenced in the Online Boutique manifest
# to its SHA256 content digest and persist a pinned manifest. Required by §3.10.
#
# Behaviour:
#   1. Reads the cached upstream manifest kubernetes-manifests-v0.10.5.yaml.
#   2. For each unique image:tag reference, resolves the MANIFEST-LIST digest
#      directly from the registry via `docker buildx imagetools inspect` — no
#      image pull, and the digest is platform-independent (multi-arch), so the
#      same pinned manifest is valid on the arm64 dev Mac and the x86_64
#      measurement VM alike.
#   3. Writes pinned-manifest.yaml — same content, but every `image: foo:tag`
#      becomes `image: foo:tag@sha256:...` (tag kept for readability; the
#      runtime resolves by digest).
#   4. Writes pinned-digests.txt — flat list, one image per line, suitable for
#      pasting into the §3.10 reproducibility-package appendix.
#
# Re-run after any upstream manifest update. Outputs are committed to git.
#
set -euo pipefail

OB_VERSION="v0.10.5"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_MANIFEST="$SCRIPT_DIR/kubernetes-manifests-${OB_VERSION}.yaml"
PINNED_MANIFEST="$SCRIPT_DIR/pinned-manifest.yaml"
DIGESTS_FILE="$SCRIPT_DIR/pinned-digests.txt"

if [[ ! -f "$SRC_MANIFEST" ]]; then
  echo "❌  Source manifest not found: $SRC_MANIFEST"
  echo "    Run 'bash infra/online-boutique/install.sh' first to fetch it."
  exit 1
fi

# Extract every unique `image:` line (strip whitespace and the leading 'image:').
IMAGES=$(grep -E '^[[:space:]]*image:[[:space:]]*' "$SRC_MANIFEST" \
  | sed -E 's/^[[:space:]]*image:[[:space:]]*//' \
  | tr -d '"' \
  | sort -u)

echo "▶  Found $(echo "$IMAGES" | wc -l | tr -d ' ') unique image references"

# Begin pinned manifest from source.
cp "$SRC_MANIFEST" "$PINNED_MANIFEST"
: > "$DIGESTS_FILE"

echo "# Online Boutique ${OB_VERSION} — pinned image digests (§3.10)"      >> "$DIGESTS_FILE"
echo "# Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by pin-images.sh"          >> "$DIGESTS_FILE"
echo "# Format: <original image:tag>  <repo>@sha256:<digest>"                >> "$DIGESTS_FILE"
echo ""                                                                       >> "$DIGESTS_FILE"

FAILED=0
while IFS= read -r IMG; do
  [[ -z "$IMG" ]] && continue
  echo "▶  Resolving: $IMG"

  # Manifest-list digest straight from the registry (no pull). JSON output is
  # parsed with python because the human template renders multi-line for
  # images carrying attestation manifests (e.g. busybox). MediaType is
  # recorded so the multi-arch property is verifiable in the committed output.
  MANIFEST_JSON=$(docker buildx imagetools inspect "$IMG" --format '{{json .Manifest}}' 2>/dev/null || true)
  HASH=$(printf '%s' "$MANIFEST_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("digest",""))' 2>/dev/null || true)
  MEDIA=$(printf '%s' "$MANIFEST_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mediaType",""))' 2>/dev/null || true)

  if ! printf '%s' "$HASH" | grep -qE '^sha256:[0-9a-f]{64}$'; then
    echo "❌  Could not resolve digest for $IMG (got: '$HASH')"
    FAILED=1
    continue
  fi

  DIGEST="${IMG}@${HASH}"
  echo "   → ${HASH}  (${MEDIA})"
  echo "$IMG  $DIGEST  $MEDIA" >> "$DIGESTS_FILE"

  # In-place rewrite the manifest. Escape slashes / dots for sed.
  ESC_IMG=$(printf '%s' "$IMG" | sed 's/[\/&]/\\&/g')
  ESC_DIG=$(printf '%s' "$DIGEST" | sed 's/[\/&]/\\&/g')
  # Match the whole line so we don't substring-match accidentally.
  sed -i.bak -E "s|(^[[:space:]]*image:[[:space:]]*)\"?${ESC_IMG}\"?[[:space:]]*$|\1${ESC_DIG}|g" "$PINNED_MANIFEST"
  rm -f "${PINNED_MANIFEST}.bak"
done <<< "$IMAGES"

# Verify: any unpinned `image:` lines remaining?
REMAINING=$(grep -E '^[[:space:]]*image:[[:space:]]*' "$PINNED_MANIFEST" | grep -v '@sha256:' || true)
if [[ -n "$REMAINING" || "$FAILED" == "1" ]]; then
  echo "❌  Some images remain unpinned in $PINNED_MANIFEST:"
  echo "$REMAINING"
  exit 1
else
  echo ""
  echo "✅  Every image in $PINNED_MANIFEST is now digest-pinned."
fi

echo ""
echo "Wrote:"
echo "  - $PINNED_MANIFEST"
echo "  - $DIGESTS_FILE"
echo ""
echo "Commit both to git. For bit-exact runs, re-apply with:"
echo "  kubectl apply -f $PINNED_MANIFEST"
