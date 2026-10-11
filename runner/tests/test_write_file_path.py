"""PB write_file collapses any agent-supplied path to a FLAT basename inside workspace/output, because every
PhysicianBench checkpoint reads read_output_file(OUTPUT_DIR/<flat name>). A handler that kept sub-directories would
nest a file the agent wrote as /mnt/data/x.md or /tmp/x.md, and the flat checkpoint read would grade it 'Output file
not found'. The same function also keeps every name inside workspace/output (no traversal)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments import _safe_output_name


def test_conventional_and_prefixed_paths_flatten():
    assert _safe_output_name("management_plan.txt") == "management_plan.txt"
    assert _safe_output_name("workspace/output/management_plan.txt") == "management_plan.txt"
    assert _safe_output_name("output/management_plan.txt") == "management_plan.txt"
    assert _safe_output_name("workspace/management_plan.txt") == "management_plan.txt"


def test_absolute_and_sandbox_paths_flatten_to_basename():
    # absolute and sandbox-style paths agents write to (sandbox mounts, /tmp, home directories)
    assert _safe_output_name("/mnt/data/tb_workup_assessment.md") == "tb_workup_assessment.md"
    assert _safe_output_name("/tmp/id_consult_note.md") == "id_consult_note.md"
    assert _safe_output_name("/Users/Shared/tb_workup_assessment.md") == "tb_workup_assessment.md"
    assert _safe_output_name("mnt/data/tb_workup_assessment.md") == "tb_workup_assessment.md"


def test_traversal_is_blocked():
    # basename can never escape workspace/output
    assert _safe_output_name("../../etc/passwd") == "passwd"
    assert _safe_output_name("/etc/passwd") == "passwd"
    assert _safe_output_name("..") is None
    assert _safe_output_name("") is None
    assert _safe_output_name("/") is None
