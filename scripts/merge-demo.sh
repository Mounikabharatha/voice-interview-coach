#!/usr/bin/env bash
# Merge two (or more) demo recordings into one file sized for GitHub.
#
#   ./scripts/merge-demo.sh part1.mp4 part2.mp4
#
# Joins and compresses in a SINGLE ffmpeg pass. Merging first and compressing afterwards
# re-encodes the video twice, and the second pass compounds the artefacts of the first.
#
# Re-encoding rather than stream-copying is deliberate: the concat demuxer needs identical
# codecs, resolution and timebase across inputs, which two separate screen recordings often
# do not have. This normalises everything and is slower but never fails on a mismatch.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <first.mp4> <second.mp4> [more.mp4 ...]" >&2
  exit 1
fi

command -v ffmpeg >/dev/null || { echo "ffmpeg not found. Run: brew install ffmpeg" >&2; exit 1; }

OUT="demo-merged.mp4"
TARGET_MB=10          # GitHub's per-file limit in a README

for f in "$@"; do
  [ -f "$f" ] || { echo "no such file: $f" >&2; exit 1; }
done

echo "inputs:"
for f in "$@"; do
  printf '  %-45s %s\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done

# Build the filter: scale each input to a common width, then concat video+audio together.
n=$#
inputs=(); filters=""; streams=""
i=0
for f in "$@"; do
  inputs+=(-i "$f")
  # setsar keeps pixel aspect consistent; two recordings can otherwise differ and concat fails
  filters+="[${i}:v]scale=1280:-2,setsar=1,fps=24[v${i}];[${i}:a]aresample=48000[a${i}];"
  streams+="[v${i}][a${i}]"
  i=$((i + 1))
done
filters+="${streams}concat=n=${n}:v=1:a=1[outv][outa]"

echo
echo "merging and compressing in one pass..."
ffmpeg -y -hide_banner -loglevel error -stats \
  "${inputs[@]}" \
  -filter_complex "$filters" -map "[outv]" -map "[outa]" \
  -c:v libx264 -crf 30 -preset slow -pix_fmt yuv420p \
  -c:a aac -b:a 96k -movflags +faststart \
  "$OUT"

SIZE_MB=$(( $(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT") / 1024 / 1024 ))
echo
echo "wrote $OUT  (${SIZE_MB} MB)"

if [ "$SIZE_MB" -ge "$TARGET_MB" ]; then
  echo
  echo "That is over GitHub's ${TARGET_MB} MB limit. Re-run the last command with:"
  echo "   -crf 34          (more compression)"
  echo "   scale=1024:-2    (smaller frame)"
else
  echo "Under GitHub's ${TARGET_MB} MB limit — ready to drag into the README."
fi
