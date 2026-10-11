#!/usr/bin/env bash
# Reset the PhysicianBench FHIR server to its pristine state: kill the FHIR server on the configured port
# (by PID), re-extract the PRISTINE H2 database from the OCI layer, then start a clean server. Also used to
# recover from a hot-copy corruption.
set -e
ROOT="${MH_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
H2PARENT="${MH_H2_PARENT:-$ROOT/benchmark_dataprocess/PhysicianBench/h2data}"
H2DIR="$H2PARENT/tmp"
LAYER="$ROOT/benchmark/PhysicianBench/physicianbench-fhir-v1/blobs/sha256/56102fee37eca8384bdd97ae6cc66ac27c4efae8b85f85341166197ff39ea64d"

echo "== kill servers by PID =="
ps -u "$USER" -o pid,cmd 2>/dev/null | grep "[m]ain.war" | grep "server.port=${MH_FHIR_PORT:-38080}" | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
# wait for the killed server to be gone: poll every 0.2 s, at most 5 s
for _w in $(seq 1 25); do ps -u "$USER" -o cmd 2>/dev/null | grep "[m]ain.war" | grep -q "server.port=${MH_FHIR_PORT:-38080}" || break; sleep 0.2; done
ps -u "$USER" -o pid,cmd 2>/dev/null | grep "[m]ain.war" | grep -v grep || echo "no fhir procs"

echo "== re-extract pristine H2 =="
rm -f "$H2DIR"/hapi-h2db.mv.db "$H2DIR"/*.lock.db "$H2DIR"/hapi-h2db.trace.db 2>/dev/null || true
mkdir -p "$H2DIR"
tar xf "$LAYER" -C "$H2PARENT" tmp/hapi-h2db.mv.db
ls -la "$H2DIR/hapi-h2db.mv.db"

echo "== start clean server =="
bash "$ROOT/benchmark_dataprocess/PhysicianBench/run_fhir.sh" "${MH_FHIR_PORT:-38080}" >/dev/null 2>&1

echo "== wait for Started Application =="
LOG="${MH_FHIR_LOG:-$ROOT/benchmark_dataprocess/PhysicianBench/fhir-server.log}"
for i in $(seq 1 180); do                          # poll every 1 s, up to 180 s
  grep -q "Started Application in" "$LOG" 2>/dev/null && { echo "ready after ${i}x1s"; break; }
  sleep 1
done

echo "== strict ready gate (fail-closed) =="
ready=0
for i in $(seq 1 120); do                          # poll every 1 s, up to 120 s
  if curl -fsS -m 10 "http://localhost:${MH_FHIR_PORT:-38080}/fhir/metadata" -H "Accept: application/fhir+json" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then echo "[FATAL] FHIR did not become ready after restore_pristine"; exit 1; fi
echo "== health (ready) =="
curl -fsS -m 10 "http://localhost:${MH_FHIR_PORT:-38080}/fhir/Patient?_summary=count" -H "Accept: application/fhir+json" || { echo "[FATAL] FHIR health probe failed"; exit 1; }
echo
