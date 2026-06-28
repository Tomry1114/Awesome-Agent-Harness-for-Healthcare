#!/usr/bin/env bash
# One-time recovery: kill all FHIR servers (by PID), re-extract PRISTINE H2 from the OCI layer,
# then start a clean server. Use after a hot-copy corruption.
set -e
ROOT="$HOME/Medical_harness"
H2DIR="$ROOT/benchmark_dataprocess/PhysicianBench/h2data/tmp"
LAYER="$ROOT/benchmark/PhysicianBench/physicianbench-fhir-v1/blobs/sha256/56102fee37eca8384bdd97ae6cc66ac27c4efae8b85f85341166197ff39ea64d"

echo "== kill servers by PID =="
ps -u "$USER" -o pid,cmd 2>/dev/null | grep "[m]ain.war" | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 5
ps -u "$USER" -o pid,cmd 2>/dev/null | grep "[m]ain.war" | grep -v grep || echo "no fhir procs"

echo "== re-extract pristine H2 =="
rm -f "$H2DIR"/hapi-h2db.mv.db "$H2DIR"/*.lock.db "$H2DIR"/hapi-h2db.trace.db 2>/dev/null || true
tar xf "$LAYER" -C "$ROOT/benchmark_dataprocess/PhysicianBench/h2data" tmp/hapi-h2db.mv.db
ls -la "$H2DIR/hapi-h2db.mv.db"

echo "== start clean server =="
bash "$ROOT/benchmark_dataprocess/PhysicianBench/run_fhir.sh" 38080 >/dev/null 2>&1

echo "== wait for Started Application =="
LOG="$ROOT/benchmark_dataprocess/PhysicianBench/fhir-server.log"
for i in $(seq 1 60); do
  grep -q "Started Application in" "$LOG" 2>/dev/null && { echo "ready after ${i}x3s"; break; }
  sleep 3
done

echo "== strict ready gate (fail-closed) =="
ready=0
for i in $(seq 1 40); do
  if curl -fsS -m 10 "http://localhost:38080/fhir/metadata" -H "Accept: application/fhir+json" >/dev/null 2>&1; then ready=1; break; fi
  sleep 3
done
if [ "$ready" -ne 1 ]; then echo "[FATAL] FHIR did not become ready after restore_pristine"; exit 1; fi
echo "== health (ready) =="
curl -fsS -m 10 "http://localhost:38080/fhir/Patient?_summary=count" -H "Accept: application/fhir+json" || { echo "[FATAL] FHIR health probe failed"; exit 1; }
echo
