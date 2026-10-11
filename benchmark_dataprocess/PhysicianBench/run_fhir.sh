#!/usr/bin/env bash
# Launch the PhysicianBench HAPI FHIR server (Singularity container, setuid install).
# Usage: ./run_fhir.sh [PORT]   (default 38080)
set -e
ROOT="${MH_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SIF="$ROOT/benchmark/PhysicianBench/fhir-full.sif"
H2="${MH_H2_PARENT:-$ROOT/benchmark_dataprocess/PhysicianBench/h2data}/tmp"
LOG="${MH_FHIR_LOG:-$ROOT/benchmark_dataprocess/PhysicianBench/fhir-server.log}"
PORT="${1:-${MH_FHIR_PORT:-38080}}"

module load singularity-ce-4.1.3 2>/dev/null || true
# `module load` is best-effort: on some cluster compute nodes it silently leaves PATH untouched (module
# is a shell function, the load prints nothing, and singularity is still not on PATH). nohup would then
# exit with "failed to run command 'singularity'" and leave a tiny log, which reads exactly like a
# server that started and refused connections. Resolve an absolute path, and fail loudly if there is none.
SING="${SINGULARITY:-$(command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null || true)}"
if [ -z "$SING" ]; then
  echo "[FATAL] singularity not found: module load left PATH untouched and no install path is executable" >&2
  exit 127
fi
# stop any running instance on this port — kill by PID via ps+awk (pkill -f is unreliable on some nodes),
# then poll every 0.2 s (at most 3 s) until it is gone
ps -u "$USER" -o pid,cmd 2>/dev/null | grep "[m]ain.war" | grep "server.port=${MH_FHIR_PORT:-38080}" | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
for _w in $(seq 1 15); do ps -u "$USER" -o cmd 2>/dev/null | grep "[m]ain.war" | grep -q "server.port=${MH_FHIR_PORT:-38080}" || break; sleep 0.2; done   # wait (up to 3 s) until the previous server process has exited

# --no-mount tmp + bind writable h2data over /tmp (image db lives at /tmp/hapi-h2db);
# --pwd /app required (Spring -Dloader.path uses paths relative to /app).
nohup "$SING" run --no-mount tmp --pwd /app --bind "$H2:/tmp" "$SIF" \
  --spring.config.location=/configs/application.yaml --server.port="$PORT" \
  > "$LOG" 2>&1 &
echo "FHIR server starting on port $PORT (PID $!)"
echo "log: $LOG"
echo "test: curl -s http://localhost:$PORT/fhir/metadata -H 'Accept: application/fhir+json'"
