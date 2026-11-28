#!/bin/bash
# Move the inclusionAI/Ling-3.2-air HF blobs onto the NVMe drive, leaving
# symlinks in place (files may be plain blobs in the repo dir or shards under
# hub/blobs/<xx>/). Only moves physical files not referenced by any other
# cached repo.
set -euo pipefail

HUB="$HOME/.cache/huggingface/hub"
SNAP=$(ls -d "$HUB/models--inclusionAI--Ling-3.2-air/snapshots"/*)
DEST=/Volumes/NVME/hf-cache/ling-3.2-air-blobs
mkdir -p "$DEST"

# physical targets of this snapshot, deduped
: > /tmp/ling-blob-list.txt
for f in "$SNAP"/*; do
  [ -L "$f" ] || continue
  realpath "$f" >> /tmp/ling-blob-list.txt
done
sort -u /tmp/ling-blob-list.txt -o /tmp/ling-blob-list.txt

# sharing check: does any OTHER repo's blobs dir link this file name?
moved=0; shared=0; skipped=0
while read -r blob; do
  name=$(basename "$blob")
  dest="$DEST/$name"
  # already moved earlier? just make sure the link is in place
  if [ -L "$dest" ]; then skipped=$((skipped+1)); continue; fi
  others=$(find "$HUB"/models--*/blobs -maxdepth 2 -lname "*/$name" 2>/dev/null \
           | grep -v "models--inclusionAI--Ling-3.2-air/" | head -1 || true)
  if [ -n "$others" ]; then
    echo "SHARED, leaving on internal: $name <- $others"
    shared=$((shared+1)); continue
  fi
  if [ ! -e "$dest" ]; then
    rsync -aL "$blob" "$dest.tmp"   # -L: copy what the symlink points to
    if ! cmp -s "$blob" "$dest.tmp"; then echo "VERIFY FAIL $name" >&2; rm -f "$dest.tmp"; exit 1; fi
    mv "$dest.tmp" "$dest"
  fi
  rm "$blob"
  ln -s "$dest" "$blob"
  moved=$((moved+1))
  echo "moved $name ($(du -h "$dest" | cut -f1))"
done < /tmp/ling-blob-list.txt

echo "DONE moved=$moved shared=$shared preexisting=$skipped"
du -sh "$DEST"
