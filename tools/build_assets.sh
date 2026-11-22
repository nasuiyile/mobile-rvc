#!/usr/bin/env bash
# Builds the base model pack bundled into the APK:
#   build/apk_assets/rvc/{hubert,fcpe,synth}
# Usage: tools/build_assets.sh [out_dir]      (PYTHON=... to pick an interpreter)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build/apk_assets/rvc}"
PY="${PYTHON:-python3}"
HUBERT="${HUBERT_DIR:-$ROOT/build/hubert_base}"
HF="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base"

if [ ! -f "$HUBERT/pytorch_model.bin" ]; then
  echo "[assets] downloading RVC hubert_base (Transformers format, ~190 MB)"
  mkdir -p "$HUBERT"
  for f in config.json preprocessor_config.json pytorch_model.bin; do
    curl -fL --retry 3 -o "$HUBERT/$f" "$HF/$f"
  done
fi

TMP="$OUT.tmp"
rm -rf "$TMP" && mkdir -p "$TMP"
cd "$ROOT/tools"
"$PY" -W ignore -m rvc_mobile.export_hubert --hubert "$HUBERT" --out "$TMP"
"$PY" -W ignore -m rvc_mobile.export_fcpe --out "$TMP"
"$PY" -W ignore -m rvc_mobile.export_synth --out "$TMP"
rm -rf "$OUT" && mv "$TMP" "$OUT"
echo "[assets] done: $OUT ($(du -sh "$OUT" | cut -f1))"
# Optional base pack for APKs built without assets
if [ "${MAKE_ZIP:-0}" = "1" ]; then
  (cd "$OUT/.." && rm -f rvc_base_pack.zip && zip -qr -0 rvc_base_pack.zip "$(basename "$OUT")")
  echo "[assets] base pack: $(dirname "$OUT")/rvc_base_pack.zip"
fi
