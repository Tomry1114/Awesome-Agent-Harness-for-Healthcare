#!/usr/bin/env bash
# Launch the PhysicianBench HAPI FHIR server (HPC ce483, Singularity setuid).
# Usage: ./run_fhir.sh [PORT]   (default 38080)
set -e
ROOT="$HOME/Medical_harness"
SIF="$ROOT/benchmark/PhysicianBench/fhir-full.sif"
H2="$ROOT/benchmark_dataprocess/PhysicianBench/h2data/tmp"
LOG="$ROOT/benchmark_dataprocess/PhysicianBench/fhir-server.log"
PORT="${1:-38080}"

module load singularity-ce-4.1.3 2>/dev/null || true
# stop any running instance — pkill -f is unreliable on this node, kill by PID via ps+awk
ps -u "$USER" -o pid,cmd 2>/dev/null | grep "[m]ain.war" | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 3

# --no-mount tmp + bind writable h2data over /tmp (image db lives at /tmp/hapi-h2db);
# --pwd /app required (Spring -Dloader.path uses paths relative to /app).
nohup singularity run --no-mount tmp --pwd /app --bind "$H2:/tmp" "$SIF" \
  --spring.config.location=/configs/application.yaml --server.port="$PORT" \
  > "$LOG" 2>&1 &
echo "FHIR server starting on port $PORT (PID $!)"
echo "log: $LOG"
echo "test: curl -s http://localhost:$PORT/fhir/metadata -H 'Accept: application/fhir+json'"
