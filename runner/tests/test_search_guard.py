# -*- coding: utf-8 -*-
"""A web search cannot reach the benchmark's own source, nor the material an item was taken from.

    runner/search_guard.py + runner/search_blocklist.yaml    the rules and the hard-wired filter
    tools_medcta.search                                        the one GoogleSearch path (refuse, search, withhold, format)
    ToolSandboxEnv.call_tool                                   puts the blocked record on the env call result
    ActionExecutor.build_event                                 lifts it onto the tool_call event
    run.py                                                     result["search_blocked"], per task, by origin

The leaked observations below are verbatim GoogleSearch results recorded in MedCTA agent trajectories. No test
reaches the network: the Serper transport (tools_medcta.google_search) is replaced, and the run.py tests go through
runner/tests/inline_loop_rig.py, which refuses every socket.
"""
import re

import pytest

import inline_loop_rig as L

PROV, DSET = "benchmark_provenance_source", "benchmark_dataset_source"

# MCTA-4, recorded agent search '"Gpc3" "E12.0" "E13.5" "E16.5" kidney': the dataset row, an ordinary embryology page, the paper.
GPT55_MCTA4 = (
    "[serper web search]\n"
    "- IVUL-KAUST/MedCTA · Datasets at Hugging Face: Compare Gpc3 +/+ and Gpc3 −/ embryonic kidneys across E12.0, E13.5, "
    "and E16.5, focusing on ureteric bud branching, kidney size, and cortical–medullary ...\n"
    "- Developmental chronology of mouse embryo from 2-cell ...: ... E16.5. E16.0. E15.5. E14.5. E14.0. E13.5. E13.0. "
    "E12.5. E12.0. E11.5. E11.0. E10.5. E10 ... Gpc3–Unc5c. Gpc3–Igf1r. Gpi1–Amfr. Grn–Sort1. Grn–Tnfrsf1a. Jag1– ...\n"
    "- MedCTA: A Benchmark for Clinical Tool Agents: Compare Gpc3 +/+ and Gpc3 -/ embryonic kidneys across E12.0, E13.5, "
    "and E16.5, focusing on ureteric bud branching, kidney size, and cortical ...")

# MCTA-54, a recorded HARNESS search (origin recovery): four ordinary medical pages and the paper quoting the reference trace.
KIMI_MCTA54_QUERY = "sentinel node metastases detection methods H&E CK-18 CEA hTRT MUC-1 Total figure"
KIMI_MCTA54 = (
    "[serper web search]\n"
    "- Micrometastases in sentinel nodes of gastric cancer - PMC: by H Ajisaka · 2003 · Cited by 91 — CK-18 mRNA was "
    "expressed in 15 SNs (7%), CEA in 12 (6%), hTRT in 10 (5%), and MUC-1 in 12 (6%). At least one mRNA marker was "
    "expressed in 25 nodes (12%) ...\n"
    "- MedCTA: A Benchmark for Clinical Tool Agents: CK-18, CEA, hTRT, and MUC-1 are commonly detected using molecular "
    "techniques such as RT-PCR or related nucleic acid-based assays to identify.\n"
    "- A clinical diagnostic test on the detection of sentinel lymph node ...: by J Shang · 2022 · Cited by 4 — RT-PCR "
    "detection can not only rapidly assess sentinel lymph node status in breast cancer patients during surgery, but its "
    "accuracy is also close to that of IHC.\n"
    "- Sentinel Lymph Node Biopsy - NCI: A sentinel lymph node biopsy (SLNB) is a procedure in which the sentinel lymph "
    "node is identified, removed, and examined to determine whether ...\n"
    "- NIH Public Access: Objective—To describe the incidence of low-volume, ultrastage-detected metastases in sentinel "
    "lymph nodes (SLNs) identified at surgical staging for ...")
KIMI_MCTA54_KEPT = "\n".join(l for l in KIMI_MCTA54.split("\n") if "MedCTA" not in l)
# on MCTA-54 itself the first line, the item's source article (Ajisaka 2003, the item's Figure 2), is withheld too
KIMI_MCTA54_KEPT_ON_ITS_TASK = "\n".join(l for l in KIMI_MCTA54_KEPT.split("\n") if "Micrometastases in sentinel" not in l)

# MCTA-11, recorded agent search: the dataset row whose text is the gold whitelist answer.
GPT55_MCTA11 = ("[serper web search]\n- IVUL-KAUST/MedCTA · Datasets at Hugging Face: Identify the key abnormalities "
                "present prior to hepatic resection. Before hepatic resection, the CT scan demonstrates a subcapsular "
                "hematoma of the right ...")

WITHHELD = ("[search guard] %d result(s) withheld: they point at the benchmark's own source (its dataset, paper or a "
            "copy of its rows).")
REFUSAL = ("[search refused] this query targets the benchmark's own source (its dataset or paper), which search may "
           "not reach; no results returned.")


@pytest.fixture
def TM(monkeypatch):
    """tools_medcta with the Serper transport replaced: a test sets TM.responses[query]; every call is recorded."""
    import adapters.medcta.tools_medcta as tm
    tm.responses, tm.transport_calls = {}, []

    def google_search(query):
        tm.transport_calls.append(query)
        return tm.responses[query]
    monkeypatch.setattr(tm, "google_search", google_search)
    yield tm
    del tm.responses, tm.transport_calls


def _organic(title, snippet, link):
    return {"title": title, "snippet": snippet, "link": link}


def _native_format(resp, q):
    """The observation format google_search produces without the guard, for comparison."""
    out = []
    ab = resp.get("answerBox") or {}
    if ab.get("answer") or ab.get("snippet"):
        out.append("[answer] " + str(ab.get("answer") or ab.get("snippet")))
    kg = resp.get("knowledgeGraph") or {}
    if kg.get("description"):
        out.append("[kg] " + str(kg.get("description")))
    for r in (resp.get("organic") or [])[:5]:
        out.append("- %s: %s" % (r.get("title", "")[:80], (r.get("snippet") or "")[:220]))
    return ("[serper web search]\n" + "\n".join(out)) if out else "[serper: no results] " + q


# Without a task id only the benchmark-level rules apply, so all eight stay. (The first is MCTA-11's source article;
# on MCTA-11's task it is withheld as item provenance, see the item-provenance tests below.)
ORDINARY = [
    _organic("Posterior Hepatic Duct Injury during Laparoscopic Cholecystectomy ... - PMC",
             "Computed tomography before hepatic resection demonstrating a subcapsular hematoma of the right hepatic lobe "
             "and an abscess in the subsegment.", "https://pmc.ncbi.nlm.nih.gov/articles/PMC2425787/"),
    _organic("CT angiography (CTA) | Radiology Reference Article | Radiopaedia.org",
             "CT angiography (CTA) is a CT technique used to visualise arterial and venous vessels.",
             "https://radiopaedia.org/articles/ct-angiography"),
    _organic("Coronary CTA in the emergency department - Med Clin North Am",
             "Coronary computed tomography angiography (CTA) for acute chest pain; OCR of the report was not used.",
             "https://www.medical.theclinics.com/article/S0025-7125(15)00050-2/fulltext"),
    _organic("flaviagiammarino/path-vqa · Datasets at Hugging Face",
             "What is present? ... yes. Is this a lymphocyte? ... no.",
             "https://huggingface.co/datasets/flaviagiammarino/path-vqa"),
    _organic("IVUL-KAUST/VideoLLM · GitHub", "Image and Video Understanding Lab video language model code.",
             "https://github.com/IVUL-KAUST/VideoLLM"),
    _organic("[2606.11703] Radiology report generation with retrieval", "We study chest X-ray report generation.",
             "https://arxiv.org/abs/2606.11703"),
    _organic("Glypican-3-deficient Mice Exhibit Developmental Overgrowth", "by DF Cano-Gauci · 1999 — kidney sections "
             "from Gpc3 mice at E12.5 and E16.5.", "https://www.semanticscholar.org/paper/Glypican-3-deficient-mice/abc123"),
    _organic("Ann Surg Oncol 2026.11702 supplement", "Sentinel lymph node staging.",
             "https://doi.org/10.1245/s10434-026-11702-x"),
]


# ------------------------------------------------------------------------------------------------ results
def test_leaked_stored_results_are_removed_and_the_ordinary_medical_result_is_kept(TM):
    q = '"Gpc3" "E12.0" "E13.5" "E16.5" kidney'
    TM.responses[q] = GPT55_MCTA4
    out, blocked = TM.search(q)
    assert out == ("[serper web search]\n" + GPT55_MCTA4.split("\n")[2] + "\n" + WITHHELD % 2)
    assert blocked == {"count": 2, "patterns": ["benchmark_name", "hf_dataset_id", "paper_title"], "what": "results",
                       "reasons": {DSET: 2}}

    TM.responses[KIMI_MCTA54_QUERY] = KIMI_MCTA54
    out, blocked = TM.search(KIMI_MCTA54_QUERY)
    assert out == KIMI_MCTA54_KEPT + "\n" + WITHHELD % 1
    assert blocked == {"count": 1, "patterns": ["benchmark_name", "paper_title"], "what": "results", "reasons": {DSET: 1}}

    q11 = '"Identify the key abnormalities present prior to hepatic resection" liver CT 2a 2b'
    TM.responses[q11] = GPT55_MCTA11
    out, blocked = TM.search(q11)
    assert out == "[serper: no results] " + q11 + "\n" + WITHHELD % 1     # every result withheld
    assert blocked["count"] == 1 and "hf_dataset_id" in blocked["patterns"]
    assert TM.transport_calls == [q, KIMI_MCTA54_QUERY, q11]


def test_live_results_are_matched_by_link_and_ordinary_medical_pages_are_not_blocked(TM):
    import search_guard as G
    leaked = [
        _organic("train-00000-of-00001.parquet", "Parquet file, 2.1 MB.",
                 "https://huggingface.co/datasets/IVUL-KAUST/MedCTA/resolve/main/data/train.parquet"),
        _organic("Overview of a clinical tool-use agents benchmark", "alphaXiv discussion.",
                 "https://www.alphaxiv.org/overview/2606.11702v1"),
        _organic("rows", "{\"row_idx\": 0}",
                 "https://datasets-server.huggingface.co/rows?dataset=IVUL-KAUST%2FMedCTA&config=default&split=train"),
        _organic("Project page", "Evaluating multimodal medical agents.", "https://ivul-kaust.github.io/MedCTA/"),
        _organic("Code", "Repository.", "https://github.com/IVUL-KAUST/MedCTA"),
    ]
    for e in leaked:
        assert G.entry_matches("MedCTA", dict(e, text="- %s: %s" % (e["title"], e["snippet"]))), e["link"]
    for e in ORDINARY:
        assert G.entry_matches("MedCTA", dict(e, text="- %s: %s" % (e["title"], e["snippet"]))) == [], e["link"]

    q = "hepatic resection subcapsular hematoma CT"
    resp = {"organic": [ORDINARY[0], leaked[0], ORDINARY[1], leaked[1], ORDINARY[2]]}
    TM.responses[q] = resp
    out, blocked = TM.search(q)
    kept = {"organic": [ORDINARY[0], ORDINARY[1], ORDINARY[2]]}
    assert out == _native_format(kept, q) + "\n" + WITHHELD % 2
    assert blocked == {"count": 2, "patterns": ["hf_dataset_url", "paper_arxiv_id_url"], "what": "results",
                       "reasons": {DSET: 2}}

    # nothing withheld: exactly the native observation, from a live response or from a recorded string at the seam
    q2 = "coronary CTA chest pain"
    TM.responses[q2] = {"answerBox": {"snippet": "CTA visualises vessels."}, "organic": ORDINARY[:5]}
    assert TM.search(q2) == (_native_format(TM.responses[q2], q2), None)
    TM.responses["x"] = "[search error] URLError('timed out')"
    assert TM.search("x") == ("[search error] URLError('timed out')", None)


# ------------------------------------------------------------------------------------------------ queries
@pytest.mark.parametrize("query,patterns", [
    ('site:huggingface.co "IVUL-KAUST/MedCTA"', ["benchmark_name", "dataset_owner"]),              # recorded agent query, MCTA-95
    ('"MedCTA" "Identify the organ shown and state whether this anatomical variant is common or rare."',
     ["benchmark_name"]),                                                                            # recorded agent query, MCTA-5
    ('"determine the condition of the bowel" PathVQA OR PMC-VQA OR MedCTA', ["benchmark_name"]),     # recorded agent query, MCTA-74
    ("A Benchmark for Clinical Tool Agents arxiv", ["paper_title"]),
    ("arXiv 2606.11702", ["paper_arxiv_id"]),
])
def test_a_benchmark_targeting_query_is_refused_without_searching(TM, query, patterns):
    out, blocked = TM.search(query)
    assert out == REFUSAL
    assert blocked == {"count": 1, "patterns": patterns, "what": "query", "reasons": {DSET: 1}}
    assert TM.transport_calls == []


@pytest.mark.parametrize("query", ["portal vein thrombosis CT findings", "head and neck CTA protocol Med Clin",
                                   "tool agents in clinical practice review", "sentinel lymph node CK-18 RT-PCR", "clinical decision tools"])
def test_an_ordinary_medical_query_is_not_refused(TM, query):
    TM.responses[query] = "[serper: no results] " + query
    assert TM.search(query) == ("[serper: no results] " + query, None)
    assert TM.transport_calls == [query]


# ------------------------------------------------------------------------------------------------ records
def test_the_env_call_result_carries_the_blocked_record(TM):
    import environments as E
    env = E.ToolSandboxEnv({"task_id": "MCTA-0", "environment": {"type": "tool_sandbox"}, "context": {"images": []}})
    TM.responses[KIMI_MCTA54_QUERY] = KIMI_MCTA54
    res = env.call_tool("GoogleSearch", {"query": KIMI_MCTA54_QUERY})
    assert res == {"tool": "GoogleSearch", "args": {"query": KIMI_MCTA54_QUERY},
                   "output": KIMI_MCTA54_KEPT + "\n" + WITHHELD % 1, "mode": "real",
                   "blocked": {"count": 1, "patterns": ["benchmark_name", "paper_title"], "what": "results",
                               "reasons": {DSET: 1}}}
    TM.responses["portal vein thrombosis"] = "[serper: no results] portal vein thrombosis"
    assert "blocked" not in env.call_tool("GoogleSearch", {"query": "portal vein thrombosis"})
    assert env.call_tool("GoogleSearch", {"query": "MedCTA dataset"})["blocked"] == {
        "count": 1, "patterns": ["benchmark_name"], "what": "query", "reasons": {DSET: 1}}
    # the env passes its task id: on MCTA-54 that item's source article is withheld as well
    env54 = E.ToolSandboxEnv({"task_id": "MCTA-54", "environment": {"type": "tool_sandbox"}, "context": {"images": []}})
    res = env54.call_tool("GoogleSearch", {"query": KIMI_MCTA54_QUERY})
    assert res["output"] == KIMI_MCTA54_KEPT_ON_ITS_TASK + "\n" + WITHHELD % 2
    assert res["blocked"]["reasons"] == {DSET: 1, PROV: 1}


def _search_reply(query):
    return '<tool_call>{"name": "GoogleSearch", "arguments": {"query": "%s"}}</tool_call>' % query.replace('"', '\\"')


def test_agent_search_blocked_record_on_the_trajectory_event_and_result_json(monkeypatch, tmp_path):
    import adapters.medcta.tools_medcta as tm
    rig = L.Rig(monkeypatch, tmp_path, "MedCTA", "MCTA-54")
    served = {KIMI_MCTA54_QUERY: KIMI_MCTA54}
    monkeypatch.setattr(tm, "google_search", lambda q: served[q])
    rig.replies = [_search_reply(KIMI_MCTA54_QUERY), _search_reply('site:huggingface.co "IVUL-KAUST/MedCTA"'),
                   "<answer>RT-PCR</answer>"]
    rig.run(max_steps=6)

    calls = rig.events("tool_call")
    assert [(e["tool"], e["origin"]) for e in calls] == [("GoogleSearch", "agent"), ("GoogleSearch", "agent")]
    assert calls[0]["blocked"] == {
        "count": 2, "patterns": ["benchmark_name", "item_byline", "item_title", "paper_title"], "what": "results",
        "reasons": {DSET: 1, PROV: 1}}
    # the record travels beside the result (ActionExecutor): not in the result, the observation or the agent's state
    assert "blocked" not in calls[0]["result"] and "blocked" not in calls[0]["observation"]
    assert not any("blocked" in (s.get("last_result") or {}) or "blocked" in str(s.get("last_observation"))
                   for s in rig.states)
    assert calls[0]["result"]["output"] == KIMI_MCTA54_KEPT_ON_ITS_TASK + "\n" + WITHHELD % 2
    assert calls[1]["blocked"] == {"count": 1, "patterns": ["benchmark_name", "dataset_owner"], "what": "query",
                                   "reasons": {DSET: 1}}
    # what the agent was shown next turn
    assert rig.model_calls[1]["messages"][-1]["content"] == "TOOL RESULT (GoogleSearch): %s" % calls[0]["result"]["output"]
    assert rig.model_calls[2]["messages"][-1]["content"] == "TOOL RESULT (GoogleSearch): " + REFUSAL
    assert rig.result["search_blocked"] == {
        "calls": 2, "results_removed": 2, "queries_refused": 1,
        "by_origin": {"agent": {"calls": 2, "results_removed": 2, "queries_refused": 1}},
        "by_reason": {DSET: {"calls": 2, "results_removed": 1, "queries_refused": 1},
                      PROV: {"calls": 1, "results_removed": 1, "queries_refused": 0}}}


def test_a_harness_acquire_search_is_filtered_and_recorded(monkeypatch, tmp_path):
    import adapters.medcta.tools_medcta as tm
    rig = L.Rig(monkeypatch, tmp_path, "MedCTA", "MCTA-54")
    monkeypatch.setattr(tm, "google_search", lambda q: {KIMI_MCTA54_QUERY: KIMI_MCTA54}[q])
    fired = []

    def before_final(answer, ctx):
        if fired:
            return None
        fired.append(answer)
        return L.decision("ACQUIRE", rule_id="scripted_acquire", reason="retrieve external knowledge",
                          extra={"next_action": {"capability": "retrieve_knowledge", "tool": "GoogleSearch",
                                                 "args": {"query": KIMI_MCTA54_QUERY}, "read_only": True},
                                 "evidence_class": "knowledge"})
    rig.hooks.before_final = before_final
    rig.replies = ["<answer>immunohistochemistry</answer>", "<answer>RT-PCR</answer>", "<answer>RT-PCR</answer>"]
    rig.run(max_steps=6)

    calls = rig.events("tool_call")
    assert len(calls) == 1 and calls[0]["origin"] != "agent" and calls[0]["tool"] == "GoogleSearch"
    assert calls[0]["blocked"] == {
        "count": 2, "patterns": ["benchmark_name", "item_byline", "item_title", "paper_title"], "what": "results",
        "reasons": {DSET: 1, PROV: 1}}
    assert "blocked" not in calls[0]["result"] and "blocked" not in calls[0]["observation"]
    shown = KIMI_MCTA54_KEPT_ON_ITS_TASK + "\n" + WITHHELD % 2
    assert calls[0]["result"]["output"] == shown
    acquired = [s.get("harness_acquired") for s in rig.states if s.get("harness_acquired")]
    assert acquired and acquired[0]["output"] == shown
    assert "MedCTA" not in "".join(m["content"] for c in rig.model_calls for m in c["messages"]
                                   if isinstance(m.get("content"), str) and m["role"] == "user")
    origin = calls[0]["origin"]
    assert rig.result["search_blocked"] == {
        "calls": 1, "results_removed": 2, "queries_refused": 0,
        "by_origin": {origin: {"calls": 1, "results_removed": 2, "queries_refused": 0}},
        "by_reason": {DSET: {"calls": 1, "results_removed": 1, "queries_refused": 0},
                      PROV: {"calls": 1, "results_removed": 1, "queries_refused": 0}}}


def test_a_medcta_run_without_blocks_records_zero(monkeypatch, tmp_path):
    rig = L.Rig(monkeypatch, tmp_path, "MedCTA", "MCTA-0")
    rig.replies = ["<answer>deep vein thrombosis</answer>"]
    rig.run(max_steps=3)
    assert rig.result["search_blocked"] == {"calls": 0, "results_removed": 0, "queries_refused": 0, "by_origin": {},
                                            "by_reason": {}}


# ------------------------------------------------------------------------------------------------ the data file
def test_the_blocklist_has_a_section_per_benchmark_and_a_malformed_one_fails_closed(tmp_path):
    import search_guard as G
    bl = G.blocklist()
    assert set(bl) == {"MedCTA", "PhysicianBench", "HealthAdminBench"}
    assert len(bl["MedCTA"]["items"]) == 107
    assert all(bl["MedCTA"][k] for k in ("refuse_query", "urls", "text"))
    assert all(bl[b][k] == [] for b in ("PhysicianBench", "HealthAdminBench") for k in ("refuse_query", "urls", "text"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("MedCTA:\n  refuse_query: []\n  urls: [{id: x, hosts: [a.org]}]\n  text: []\n")
    with pytest.raises(ValueError):
        G.blocklist(str(bad))
    with pytest.raises(KeyError):
        G.refuse_query("NoSuchBench", "anything")


# ================================================================================================ item provenance
# Item provenance: each item's direct material provenance -- the article holding its original image or figure, that
# figure or caption page, and the article's supplement -- is withheld on that item's task under blocked_reason
# benchmark_provenance_source. Other medical literature stays searchable, and another item's source is not blocked
# (the guard applies the current item's table entry only). The table is runner/search_blocklist.yaml MedCTA.items.
#
# The observations below are verbatim GoogleSearch results from recorded agent trajectories.
# MCTA-11 recorded agent search 1
GPT55_MCTA11_S1_Q = '"Identify the key abnormalities present prior to hepatic resection" liver CT 2a 2b'
GPT55_MCTA11_S1 = ('[serper web search]\n'
    '- IVUL-KAUST/MedCTA · Datasets at Hugging Face: Identify the key abnormalities present prior to hepatic resection. Before hepatic resection, the CT scan demonstrates a subcapsular hematoma of the right ...')
# MCTA-11 recorded agent search 3
GPT55_MCTA11_S3_Q = '"subcapsular hematoma of the right hepatic lobe" "abscess" "hepatic resection" CT'
GPT55_MCTA11_S3 = ('[serper web search]\n'
    '- Posterior Hepatic Duct Injury during Laparoscopic Cholecystectomy ...: Computed tomography before hepatic resection demonstrating a subcapsular hematoma of the right hepatic lobe and an abscess in the subsegment. The presence ...\n'
    '- IVUL-KAUST/MedCTA · Datasets at Hugging Face: Before hepatic resection, the CT scan demonstrates a subcapsular hematoma of the right hepatic lobe and an abscess in the subsegment. [ "OCR ...')
# MCTA-11 recorded agent search 4
GPT55_MCTA11_S4_Q = '"Computed tomography before hepatic resection demonstrating" "subcapsular hematoma" "abscess"'
GPT55_MCTA11_S4 = ('[serper web search]\n'
    '- Posterior Hepatic Duct Injury during Laparoscopic ... - PMC: by T Ohtsuka · 1999 · Cited by 19 — Computed tomography before hepatic resection demonstrating a subcapsular hematoma of the right hepatic lobe and an abscess in the subsegment. The presence ...')
# MCTA-11 recorded agent search 5
GPT55_MCTA11_S5_Q = '"Posterior Hepatic Duct Injury during Laparoscopic Cholecystectomy" "subsegment" abscess'
GPT55_MCTA11_S5 = ('[serper web search]\n'
    '- Posterior Hepatic Duct Injury during Laparoscopic Cholecystectomy ...: Posterior Hepatic Duct Injury during Laparoscopic Cholecystectomy ... abscess in the posterosuperior subsegment was observed (Figure 2b). Finally ...\n'
    '- [PDF] Posterior Hepatic Duct Injury during Laparoscopic Cholecystectomy ...: Posterior Hepatic Duct Injury during Laparoscopic Cholecystectomy ... subsegment was observed (Figure 2b). Finally, the ... capsular hematoma and abscess of the ...')
# MCTA-11 recorded agent search (a second agent), search 1
GEM3FLASH_MCTA11_S1_Q = '"2a" "2b" liver CT scan hepatic resection abnormalities'
GEM3FLASH_MCTA11_S1 = ('[serper web search]\n'
    '- Right Liver Lobe Hypoplasia and Related Abnormalities - PMC: by B Alicioglu · 2015 · Cited by 9 — The common bile duct had a more vertical course than normal (Figure 2A, 2B). The abdominal aorta and inferior vena cava were tortuous and displaced to the right ...\n'
    '- Detection of Hepatic Lesions in Candidates for Surgery: by DA Bluemke · 2000 · Cited by 115 — In the same patient, CT showed three additional lesions, but no abnormality was found on MR imaging or at surgery (Fig. 2A,2B,2C,2D). These lesions were ...\n'
    '- Pitfalls in Liver Imaging | Radiology: by V Vilgrain · 2016 · Cited by 56 — Open in Viewer Figure 2a: MR images in a 27-year-old woman with morphologic ... Open in Viewer Figure 2b: MR images in a 27-year-old woman with ...\n'
    '- Double cone-unit laparoscopic hepatic resection for tumors ...: by J Yasuda · 2023 · Cited by 1 — 2A). In case B, the tumor was in Segment4/5, and the hepatic vein close to the tumor was the V4, with two tertiary branches were G5a and G4b (Fig. 2B). ... liver ...\n'
    '- Hepatic compartment syndrome, a rare complication after ...: by A Nassar · 2024 · Cited by 2 — No injury in the liver parenchyma and hemostasis was achieved with bipolar cautery (Fig. 2A, 2B). During surgery, the patient was unstable, ...')

# MCTA-11's source article: JSLS 1999 (Ohtsuka et al.), PMC3015369, PMID 10694080; the item's image is its Figure 2.
MCTA11_SOURCE_PAGES = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3015369/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3015369/figure/F2/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3015369/bin/jsls-3-4-323-g02.jpg",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3015369/pdf/jsls-3-4-323.pdf",
    "https://europepmc.org/article/PMC/PMC3015369",
    "https://europepmc.org/abstract/MED/10694080",
    "https://pubmed.ncbi.nlm.nih.gov/10694080/",
]
# Provenance strings of MCTA-11 that nothing but the guard may hold.
MCTA11_PROVENANCE_STRINGS = ("PMC3015369", "10694080", "Ohtsuka", "Posterior Hepatic Duct Injury",
                             "Computed tomography before hepatic resection demonstrating", "Laparoscopic Cholecystectomy",
                             PROV, "item_title", "item_byline", "item_figure_caption", "item_pmcid", "item_pmid")


def _c2_organic(title, snippet, link):
    return {"title": title, "snippet": snippet, "link": link}


def test_c2_mcta11_source_article_is_removed_as_item_provenance(TM):
    # the recorded source-article results of MCTA-11 agent searches 4 and 5, on MCTA-11's task
    TM.responses[GPT55_MCTA11_S4_Q] = GPT55_MCTA11_S4
    TM.responses[GPT55_MCTA11_S5_Q] = GPT55_MCTA11_S5
    out, blocked = TM.search(GPT55_MCTA11_S4_Q, "MCTA-11")
    assert out == "[serper: no results] " + GPT55_MCTA11_S4_Q + "\n" + WITHHELD % 1
    assert blocked == {"count": 1, "patterns": ["item_byline", "item_figure_caption", "item_title"], "what": "results",
                       "reasons": {PROV: 1}}
    out, blocked = TM.search(GPT55_MCTA11_S5_Q, "MCTA-11")
    assert out == "[serper: no results] " + GPT55_MCTA11_S5_Q + "\n" + WITHHELD % 2
    assert blocked == {"count": 2, "patterns": ["item_title"], "what": "results", "reasons": {PROV: 2}}

    # live-shaped: the article, its figure page, its supplement file, its PDF, Europe PMC and PubMed copies, matched by
    # the link alone (the titles here carry nothing of the article)
    import search_guard as G
    for link in MCTA11_SOURCE_PAGES:
        e = _c2_organic("Case report", "Laparoscopic surgery.", link)
        dataset, provenance = G.entry_hits("MedCTA", dict(e, text="- Case report: Laparoscopic surgery."), "MCTA-11")
        assert dataset == [] and provenance, link
    q = "hepatic resection abscess case report"
    TM.responses[q] = {"organic": [_c2_organic("Case report", "Laparoscopic surgery.", l) for l in MCTA11_SOURCE_PAGES[:5]]}
    out, blocked = TM.search(q, "MCTA-11")
    assert out == "[serper: no results] " + q + "\n" + WITHHELD % 5
    assert blocked == {"count": 5, "patterns": ["item_pmcid"], "what": "results", "reasons": {PROV: 5}}

    # the item's provenance id in a query is refused before searching, on MCTA-11's task
    n = len(TM.transport_calls)
    assert TM.search("PMC3015369 figure 2", "MCTA-11") == (REFUSAL, {"count": 1, "patterns": ["item_pmcid"],
                                                                     "what": "query", "reasons": {PROV: 1}})
    assert TM.search("pubmed 10694080", "MCTA-11")[1]["patterns"] == ["item_pmid"]
    assert len(TM.transport_calls) == n

    # without a task id (a probe outside any task) only the benchmark-level rules apply
    assert TM.search(GPT55_MCTA11_S4_Q) == (GPT55_MCTA11_S4, None)


def test_c2_an_independent_article_on_the_same_topic_is_kept(TM):
    # MCTA-11 recorded agent search (second agent) 1, verbatim: five independent liver-imaging / hepatic-resection articles
    TM.responses[GEM3FLASH_MCTA11_S1_Q] = GEM3FLASH_MCTA11_S1
    assert TM.search(GEM3FLASH_MCTA11_S1_Q, "MCTA-11") == (GEM3FLASH_MCTA11_S1, None)

    # live-shaped on MCTA-11: an independent PMC article on the same topic (Alicioglu 2015, PMC4648108, its real PMCID,
    # seen in a recorded MCTA-11 search), and synthetic same-topic pages (bile duct injury in laparoscopic cholecystectomy
    # under a neighbouring PMC id, hepatic abscess, a same-topic DOI; the links are made up, not real records)
    import search_guard as G
    independent = [
        _c2_organic("Right Liver Lobe Hypoplasia and Related Abnormalities - PMC",
                    "by B Alicioglu · 2015 — The common bile duct had a more vertical course than normal (Figure 2A, 2B).",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4648108/"),
        _c2_organic("Bile duct injury during laparoscopic cholecystectomy - PMC",
                    "Classification and management of bile duct injuries; right posterior sectoral duct injuries.",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3015368/"),
        _c2_organic("Hepatic abscess | Radiology Reference Article | Radiopaedia.org",
                    "Subcapsular collections of the right hepatic lobe on CT.", "https://radiopaedia.org/articles/hepatic-abscess"),
        _c2_organic("Subcapsular hepatic hematoma after laparoscopic cholecystectomy",
                    "Case series; CT before hepatic resection.", "https://doi.org/10.1007/s00464-015-4301-5"),
    ]
    for e in independent:
        assert G.entry_matches("MedCTA", dict(e, text="- %s: %s" % (e["title"], e["snippet"])), "MCTA-11") == [], e["link"]
    q = "posterior hepatic duct injury laparoscopic cholecystectomy hepatic resection"
    TM.responses[q] = {"organic": independent}
    assert TM.search(q, "MCTA-11") == (_native_format({"organic": independent}, q), None)

    # ordinary topic queries on MCTA-11 run; so does a neighbouring PMC id
    for q in ("subcapsular hematoma right hepatic lobe abscess CT", "bile duct injury laparoscopic cholecystectomy review",
              "PMC3015368"):
        TM.responses[q] = "[serper: no results] " + q
        assert TM.search(q, "MCTA-11") == ("[serper: no results] " + q, None)


def test_c2_a_dataset_page_hit_keeps_its_own_reason(TM):
    # MCTA-11 recorded agent search 3: the source article and the dataset row in one observation
    TM.responses[GPT55_MCTA11_S3_Q] = GPT55_MCTA11_S3
    out, blocked = TM.search(GPT55_MCTA11_S3_Q, "MCTA-11")
    assert out == "[serper: no results] " + GPT55_MCTA11_S3_Q + "\n" + WITHHELD % 2
    assert blocked["count"] == 2 and blocked["reasons"] == {DSET: 1, PROV: 1}
    assert {"hf_dataset_id", "item_title", "item_figure_caption"} <= set(blocked["patterns"])

    # the dataset row alone on the same task: its own reason only
    TM.responses[GPT55_MCTA11_S1_Q] = GPT55_MCTA11_S1
    assert TM.search(GPT55_MCTA11_S1_Q, "MCTA-11")[1]["reasons"] == {DSET: 1}

    # a result matching both kinds counts once, under the dataset reason
    both = _c2_organic("IVUL-KAUST/MedCTA · Datasets at Hugging Face", "Search PMC3015369 in the viewer.",
                       "https://huggingface.co/datasets/IVUL-KAUST/MedCTA/viewer?q=PMC3015369")
    q = "hepatic duct injury dataset"
    TM.responses[q] = {"organic": [both]}
    out, blocked = TM.search(q, "MCTA-11")
    assert blocked["count"] == 1 and blocked["reasons"] == {DSET: 1}
    assert {"hf_dataset_url", "item_pmcid"} <= set(blocked["patterns"])

    # queries: a dataset-targeting query keeps the dataset reason on an item's task, even when it also names the id
    assert TM.search("MedCTA hepatic resection", "MCTA-11")[1] == {"count": 1, "patterns": ["benchmark_name"],
                                                                   "what": "query", "reasons": {DSET: 1}}
    assert TM.search("MedCTA PMC3015369", "MCTA-11")[1]["reasons"] == {DSET: 1}


def test_c2_another_items_provenance_is_not_blocked(TM):
    """The guard applies the current item's provenance only: MCTA-11's source article is not where MCTA-54 (or an item
    whose provenance is unknown) was taken from, and MCTA-54's source is not MCTA-11's."""
    import search_guard as G
    unknown = next(t for t, r in G.blocklist()["MedCTA"]["items"].items() if r is None)
    for tid in ("MCTA-54", unknown):
        TM.responses[GPT55_MCTA11_S4_Q] = GPT55_MCTA11_S4
        assert TM.search(GPT55_MCTA11_S4_Q, tid) == (GPT55_MCTA11_S4, None), tid
        TM.responses["PMC3015369"] = "[serper: no results] PMC3015369"
        assert TM.search("PMC3015369", tid) == ("[serper: no results] PMC3015369", None), tid
        for link in MCTA11_SOURCE_PAGES:
            assert G.entry_matches("MedCTA", {"text": "- Case report: x", "link": link}, tid) == [], (tid, link)

    # MCTA-54's recorded harness search: on MCTA-54 its source article (Ajisaka 2003) goes with the paper; on MCTA-11 only
    # the paper does
    TM.responses[KIMI_MCTA54_QUERY] = KIMI_MCTA54
    out, blocked = TM.search(KIMI_MCTA54_QUERY, "MCTA-54")
    assert out == KIMI_MCTA54_KEPT_ON_ITS_TASK + "\n" + WITHHELD % 2
    assert blocked["reasons"] == {DSET: 1, PROV: 1}
    out, blocked = TM.search(KIMI_MCTA54_QUERY, "MCTA-11")
    assert out == KIMI_MCTA54_KEPT + "\n" + WITHHELD % 1
    assert blocked == {"count": 1, "patterns": ["benchmark_name", "paper_title"], "what": "results", "reasons": {DSET: 1}}


def test_c2_the_agent_visible_note_is_generic(TM):
    """The agent sees the same withheld note and refusal whatever the reason: nothing says that a result or query was
    the item's source."""
    TM.responses[GPT55_MCTA11_S5_Q] = GPT55_MCTA11_S5
    prov_out, _ = TM.search(GPT55_MCTA11_S5_Q, "MCTA-11")
    TM.responses[GPT55_MCTA11_S1_Q] = GPT55_MCTA11_S1
    dset_out, _ = TM.search(GPT55_MCTA11_S1_Q, "MCTA-11")
    prov_note, dset_note = prov_out.rsplit("\n", 1)[1], dset_out.rsplit("\n", 1)[1]
    assert prov_note == WITHHELD % 2 and dset_note == WITHHELD % 1
    assert TM.search("PMC3015369", "MCTA-11")[0] == TM.search("MedCTA", "MCTA-11")[0] == REFUSAL
    for text in (prov_note, REFUSAL):
        low = text.lower()
        for word in ("provenance", "item", "article", "figure", "caption", "supplement", "pmc", "pubmed", "original",
                     "taken from", "image"):
            assert word not in low, (word, text)


def test_c2_every_medcta_task_has_an_item_entry_and_its_pages_are_blocked():
    import json
    import os
    import yaml
    import search_guard as G
    tasks = [json.loads(l)["task_id"] for l in open(os.path.join(L.REPO, "benchmark_dataprocess", "MedCTA",
                                                                  "tasks_unified.jsonl"), encoding="utf-8") if l.strip()]
    raw = yaml.safe_load(open(G.BLOCKLIST_PATH, encoding="utf-8"))["MedCTA"]["items"]
    compiled = G.blocklist()["MedCTA"]["items"]
    assert len(tasks) == 107 and set(raw) == set(compiled) == set(tasks)
    known = [t for t in tasks if raw[t]["status"] == "known"]
    unknown = [t for t in tasks if raw[t]["status"] == "unknown"]
    assert known and unknown and len(known) + len(unknown) == 107
    assert "MCTA-11" in known and all(raw[t].get("evidence") for t in unknown)
    for tid in known:
        entry = raw[tid]
        assert entry["evidence"] and entry["confidence"] in ("high", "medium") and entry.get("ids"), tid
        pages = entry["pages"]
        assert pages.get("article") and pages.get("figure") and pages.get("supplement"), tid
        for what, urls in pages.items():
            for url in (urls if isinstance(urls, list) else [urls]):
                dataset, provenance = G.entry_hits("MedCTA", {"text": "- x: y", "link": url}, tid)
                assert dataset == [] and provenance, (tid, what, url)
                other = unknown[0]
                assert G.entry_matches("MedCTA", {"text": "- x: y", "link": url}, other) == [], (tid, other, url)
    assert G.blocklist()["PhysicianBench"]["items"] == {} and G.blocklist()["HealthAdminBench"]["items"] == {}


@pytest.mark.parametrize("item", [
    "{status: known, evidence: [x], confidence: high}",                                   # known without an id
    "{status: known, evidence: [x], confidence: high, ids: {pmcid: '3015369'}}",           # malformed PMCID
    "{status: known, evidence: [x], confidence: high, ids: {pmid: '1'}, text: [{id: MCTA-1-title, pattern: x}]}",
    "{status: unknown}",                                                                   # unknown without evidence
    "{status: maybe, evidence: [x]}",
])
def test_c2_a_malformed_item_entry_fails_closed(tmp_path, item):
    import search_guard as G
    bad = tmp_path / "bad.yaml"
    bad.write_text("MedCTA:\n  refuse_query: []\n  urls: []\n  text: []\n  items:\n    MCTA-1: %s\n" % item)
    with pytest.raises(ValueError):
        G.blocklist(str(bad))


def test_c2_a_task_missing_from_the_item_table_fails_visibly(TM):
    with pytest.raises(KeyError):
        TM.search("portal vein thrombosis", "MCTA-9999")
    assert TM.transport_calls == []


def test_c2_the_item_table_is_not_reachable_from_prompts_or_ctx(monkeypatch, tmp_path):
    """MCTA-11 through run.run_task: the agent's search and a harness ACQUIRE search both return the item's source
    article. Nothing of the item's provenance reaches a prompt (agent or judge), the state handed to the agent, a kernel
    hook's ctx, the ledger, an observation or result.json. The only trace is the blocked record (count, generic rule
    ids, what, reasons) on the tool_call event and result.json search_blocked; ActionExecutor takes it off the env call
    result, so neither the agent's state nor the observation string nor the harness carries it. And no runtime module
    but the guard reads the blocklist."""
    import json
    import os
    import adapters.medcta.tools_medcta as tm
    judge_replies = []
    rig = L.Rig(monkeypatch, tmp_path, "MedCTA", "MCTA-11", judge=lambda p: (judge_replies.append(p), "{}")[1])
    agent_q, harness_q = "hepatic resection subcapsular hematoma abscess CT", "subsegment abscess hepatic resection"
    served = {agent_q: GPT55_MCTA11_S3.split("\n- IVUL")[0] + "\n" + GEM3FLASH_MCTA11_S1.split("\n", 1)[1],
              harness_q: GPT55_MCTA11_S4}
    monkeypatch.setattr(tm, "google_search", lambda q: served.get(q, "[serper: no results] " + q))
    seen_ctx = []
    fired = []

    def keep_ctx(*args):
        seen_ctx.append(repr(args[1]))
        try:
            seen_ctx.append(json.dumps(getattr(args[1], "__dict__", {}), default=repr))
        except Exception:
            pass
        return None

    def before_final(answer, ctx):
        keep_ctx(answer, ctx)
        if fired:
            return None
        fired.append(answer)
        return L.decision("ACQUIRE", rule_id="scripted_acquire", reason="retrieve external knowledge",
                          extra={"next_action": {"capability": "retrieve_knowledge", "tool": "GoogleSearch",
                                                 "args": {"query": harness_q}, "read_only": True},
                                 "evidence_class": "knowledge"})
    rig.hooks.before_action = keep_ctx
    rig.hooks.after_action = lambda action, ctx, result: keep_ctx(action, ctx)
    rig.hooks.before_final = before_final
    rig.replies = [_search_reply(agent_q), "<answer>a hematoma and an abscess</answer>",
                   "<answer>a hematoma and an abscess</answer>", "<answer>a hematoma and an abscess</answer>"]
    rig.run(max_steps=6)

    calls = [c for c in rig.events("tool_call") if c["tool"] == "GoogleSearch"]
    agent_call = next(c for c in calls if c["origin"] == "agent" and c["args"]["query"] == agent_q)
    harness_call = next(c for c in calls if c["origin"] != "agent" and c["args"]["query"] == harness_q)
    assert agent_call["blocked"] == {"count": 1, "patterns": ["item_figure_caption", "item_title"], "what": "results",
                                     "reasons": {PROV: 1}}
    assert harness_call["blocked"]["reasons"] == {PROV: 1}
    assert agent_call["result"]["output"] == ("[serper web search]\n" + GEM3FLASH_MCTA11_S1.split("\n", 1)[1] + "\n"
                                              + WITHHELD % 1)
    assert rig.result["search_blocked"]["by_reason"][PROV]["results_removed"] >= 2

    def strip_records(obj):
        if isinstance(obj, dict):
            return {k: strip_records(v) for k, v in obj.items() if k not in ("blocked", "search_blocked")}
        if isinstance(obj, list):
            return [strip_records(v) for v in obj]
        return obj
    surfaces = {
        "agent prompts": json.dumps(rig.model_calls, ensure_ascii=False, default=repr),
        "judge prompts": json.dumps(rig.judge_prompts + judge_replies, ensure_ascii=False, default=repr),
        "agent state": json.dumps(rig.states, ensure_ascii=False, default=repr),
        "kernel ctx": "\n".join(seen_ctx),
        "trajectory": json.dumps(strip_records(rig.trajectory), ensure_ascii=False, default=repr),
        "result.json": json.dumps(strip_records({k: v for k, v in rig.result.items() if k != "_trajectory"}),
                                  ensure_ascii=False, default=repr),
    }
    assert seen_ctx and rig.model_calls
    for where, text in surfaces.items():
        for s in MCTA11_PROVENANCE_STRINGS:
            assert s.lower() not in text.lower(), (where, s)
    for c in calls:
        if c.get("blocked"):
            assert set(c["blocked"]) == {"count", "patterns", "what", "reasons"}
            assert all(p.startswith("item_") for p in c["blocked"]["patterns"])

    # no runtime module but the guard opens the blocklist or reads its item table (comments aside). The scan also covers
    # optional top-level directories that are not shipped here; the one other allowed reader is a diff-level audit module
    # among them, which records only hashes of what it matched and reaches no prompt, ctx or run.
    readers = []
    for top in ("runner", "workspaces"):
        for root, dirs, files in os.walk(os.path.join(L.REPO, top)):
            dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
            for fn in files:
                if not fn.endswith(".py") or fn.startswith("test_"):
                    continue
                p = os.path.join(root, fn)
                code = "\n".join(line.split("#", 1)[0] for line in open(p, encoding="utf-8", errors="replace"))
                if re.search(r"search_blocklist\.yaml|BLOCKLIST_PATH|\bblocklist\(|entry_hits\(|search_guard\._item\b", code):
                    readers.append(os.path.relpath(p, L.REPO))
    assert sorted(readers) == ["runner/search_guard.py"], readers
