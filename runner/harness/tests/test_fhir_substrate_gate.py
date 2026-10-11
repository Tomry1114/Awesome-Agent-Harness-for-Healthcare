# -*- coding: utf-8 -*-
"""The substrate the agent talks to must be the substrate the readiness gate checked.

Failure mode guarded against: `--fhir-base` names one port (e.g. :38081) while restore_pristine_h2.sh boots and
health-checks ${MH_FHIR_PORT:-38080} with MH_FHIR_PORT unset. The reset's own fail-closed ready gate then passes
on the wrong port, and the agent's tool calls come back "Connection refused" while the run still completes.

Nothing downstream can tell that apart from a real result: a PB task with no reachable FHIR completes,
writes a deliverable, and gets scored. So both halves are pinned here -- one source for the port, and a
probe against the agent's own URL.
"""
import importlib.util
import os
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.dirname(os.path.dirname(_HERE))          # .../runner
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)


def _load_run_batch():
    spec = importlib.util.spec_from_file_location("_rb_under_test",
                                                  os.path.join(_RUNNER, "run_batch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = _load_run_batch()


class _Metadata(BaseHTTPRequestHandler):
    def do_GET(self):                                       # noqa: N802 -- BaseHTTPRequestHandler API
        self.send_response(200)
        self.send_header("Content-Type", "application/fhir+json")
        self.end_headers()
        self.wfile.write(b'{"resourceType":"CapabilityStatement"}')

    def log_message(self, *a):                              # keep the test output clean
        pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class OnePortOneSource(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get("MH_FHIR_PORT")
        os.environ.pop("MH_FHIR_PORT", None)

    def tearDown(self):
        os.environ.pop("MH_FHIR_PORT", None)
        if self._prev is not None:
            os.environ["MH_FHIR_PORT"] = self._prev

    def test_the_port_is_derived_from_the_url_the_agent_gets(self):
        RB._wire_fhir_port("http://localhost:38081/fhir")
        self.assertEqual(os.environ["MH_FHIR_PORT"], "38081")

    def test_a_contradicting_port_is_refused_not_silently_preferred(self):
        # The reset script would boot and verify 38080 while the agent uses 38081. Preferring either one hides
        # the mismatch, so disagreement has to stop the run.
        os.environ["MH_FHIR_PORT"] = "38080"
        with self.assertRaises(SystemExit) as cm:
            RB._wire_fhir_port("http://localhost:38081/fhir")
        self.assertIn("38080", str(cm.exception))
        self.assertIn("38081", str(cm.exception))

    def test_an_agreeing_port_is_left_alone(self):
        os.environ["MH_FHIR_PORT"] = "38081"
        RB._wire_fhir_port("http://localhost:38081/fhir")
        self.assertEqual(os.environ["MH_FHIR_PORT"], "38081")

    def test_a_base_without_a_port_is_reported_not_guessed(self):
        note = RB._wire_fhir_port("http://fhir.example.org/fhir")
        self.assertIn("no port", note)
        self.assertNotIn("MH_FHIR_PORT", os.environ)


class ProbeTheAddressUnderTest(unittest.TestCase):
    def test_a_server_that_answers_passes(self):
        port = _free_port()
        srv = HTTPServer(("127.0.0.1", port), _Metadata)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            base = "http://127.0.0.1:%d/fhir" % port
            self.assertTrue(RB._require_fhir_reachable(base, "test", attempts=5, delay=1))
        finally:
            srv.shutdown()

    def test_a_port_with_nothing_on_it_refuses_the_batch(self):
        # Not "warns": a PB task against a dead FHIR still completes and still gets scored, so a warning
        # is indistinguishable from a real run by the time anyone reads the artefacts.
        base = "http://127.0.0.1:%d/fhir" % _free_port()
        with self.assertRaises(SystemExit) as cm:
            RB._require_fhir_reachable(base, "test", attempts=2, delay=0)
        self.assertIn("REFUSING", str(cm.exception))
        self.assertIn("/metadata", str(cm.exception))

    def test_the_probe_uses_the_agents_base_url_not_an_env_port(self):
        # A gate that checks a different address than the one under test is not a gate. Serve on a real
        # port, point MH_FHIR_PORT at it, and pass a DEAD base: it must still refuse.
        port = _free_port()
        srv = HTTPServer(("127.0.0.1", port), _Metadata)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        prev = os.environ.get("MH_FHIR_PORT")
        os.environ["MH_FHIR_PORT"] = str(port)
        try:
            dead = "http://127.0.0.1:%d/fhir" % _free_port()
            with self.assertRaises(SystemExit):
                RB._require_fhir_reachable(dead, "test", attempts=2, delay=0)
        finally:
            srv.shutdown()
            os.environ.pop("MH_FHIR_PORT", None)
            if prev is not None:
                os.environ["MH_FHIR_PORT"] = prev


class TheGateIsActuallyWired(unittest.TestCase):
    def test_the_reset_call_site_probes_before_running_tasks(self):
        src = open(os.path.join(_RUNNER, "run_batch.py"), encoding="utf-8").read()
        i_wire = src.index("_wire_fhir_port(args.fhir_base)")
        i_reset = src.index('R.reset_fhir("restore_pristine")')
        i_probe = src.index("_require_fhir_reachable(args.fhir_base")
        # the task list and the resume plan (which tasks reuse a stored result) come first; the first task that runs
        # is main's run_concurrently (--workers > 1) or its own run_task call (_run_one, the workers' body, is above main)
        i_main = src.index("def main(")
        i_tasks = min(src.index("_done = run_concurrently(", i_main), src.index("res = R.run_task(", i_main))
        self.assertLess(i_wire, i_reset, "the port must be single-sourced BEFORE the reset boots a server")
        self.assertLess(i_reset, i_probe, "probe after the reset, or it can only ever see a stale server")
        self.assertLess(i_probe, i_tasks, "probe BEFORE any task runs -- afterwards the money is spent")


if __name__ == "__main__":
    unittest.main()
