#!/usr/bin/env bash
set -euo pipefail
: "${1:?usage: monitor.sh CONFIG JOBID}"
: "${2:?usage: monitor.sh CONFIG JOBID}"
CONFIG="$1"
JOBID="$2"
OUTDIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$CONFIG")"
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
S=/cm/shared/apps/slurm/current/bin
echo "== scheduler =="
"$S/squeue" -j "$JOBID" -o '%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R' || true
"$S/sacct" -j "$JOBID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode || true
echo "== output =="
if [[ -f "$OUTDIR/status.json" ]]; then cat "$OUTDIR/status.json"; else echo "status.json not written yet"; fi
echo "segments=$(find "$OUTDIR" -maxdepth 1 -name 'segment_*.extxyz' -type f 2>/dev/null | wc -l | tr -d ' ')"
if [[ -f "$OUTDIR/latest_restart.extxyz" ]]; then ls -lh "$OUTDIR/latest_restart.extxyz"; fi
echo "== recent log =="
for f in slurm-${JOBID}.out slurm-${JOBID}.err; do [[ -f "$f" ]] && { echo "--- $f"; tail -30 "$f"; }; done
