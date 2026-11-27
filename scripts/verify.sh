#!/bin/bash
# Done-gate for the psychedelic-model repo.
#   scripts/verify.sh --quick   fast tier: compile + greps + integrity (seconds)
#   scripts/verify.sh           full tier: quick + hook validation (no model download)
# A full render is NOT part of this gate (25-30 s/frame on MPS); see the manual list at the end.

set -u
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
FAIL=0

step() {
   echo ""
   echo "== $1"
}

run() {
   "$@"
   if [ $? -ne 0 ]; then
      FAIL=1
   fi
}

step "compile (syntax gate)"
run "$PY" -m compileall -q src run.py scripts/check_integrity.py

step "banned-pattern greps (comment-stripped)"
grep_gate() {
   local pattern="$1"
   local message="$2"
   shift 2
   local hits
   hits=$(for f in "$@"; do
      sed "s/#.*//" "$f" | grep -nE "$pattern" | sed "s|^|$f:|"
   done)
   if [ -n "$hits" ]; then
      echo "GATE FAIL: $message"
      echo "$hits"
      FAIL=1
   fi
}

PYFILES=$(find src run.py -name "*.py")

grep_gate "unsharp|SHARPEN|ImageEnhance" \
   "post-hoc sharpening is forbidden — it manufactures the detail we claim emerges (handoff section 3)" $PYFILES
grep_gate "torch\.float64|dtype=.float64" \
   "float64 is unsupported on MPS — keep hand-built math fp32 (CLAUDE.md section 8)" $PYFILES
grep_gate "eta\s*=\s*0?\.[0-9]*[1-9]|eta\s*=\s*[1-9]" \
   "DDIM eta > 0 is forbidden — stochastic speckle reads as fake detail" $PYFILES
grep_gate "renorm|/ *out\.norm\(\)" \
   "renormalize-to-original inside hooks is the identity for uniform gain — it nulls the knob" src/render/hooks.py

if ! grep -q "raise ValueError" src/render/engine.py || ! grep -q "BIASED_WORDS" src/render/engine.py; then
   echo "GATE FAIL: engine.py bias filter (_strip_bias / BIASED_WORDS) has been weakened or removed"
   FAIL=1
fi

step "integrity check (configs + knob directions)"
run "$PY" scripts/check_integrity.py

if [ "${1:-}" != "--quick" ]; then
   step "hook validation (tests/test_hooks.py, small U-Net, no download)"
   run env PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" tests/test_hooks.py

   step "noise drift validation (tests/test_noise_drift.py, E-8, tensor only, no download)"
   run env PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" tests/test_noise_drift.py

   step "guidance split validation (tests/test_guidance_split.py, C-3, tensor only, no download)"
   run env PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" tests/test_guidance_split.py
fi

echo ""
if [ $FAIL -ne 0 ]; then
   echo "VERIFY: FAIL"
   exit 1
fi

echo "VERIFY: PASS"
echo ""
echo "Manual checks this script cannot run:"
echo "  - render + LOOK at frames (run.py configs/crisp_trip.yaml); the score is only a proxy"
echo "  - score ranking: .venv/bin/python tests/geometry_score.py output/latest/frames"
echo "  - seed robustness before claiming any result (one magic seed is not a result)"
echo "  - arm labeling: never mix arm A (pure) and arm B (log-polar) in one claim"
exit 0
