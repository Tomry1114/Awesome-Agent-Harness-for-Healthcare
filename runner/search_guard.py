# -*- coding: utf-8 -*-
"""Search guard: a web search never returns the benchmark's own source, nor the material a benchmark item was taken
from.

Every web search of a benchmark run, whoever issues it (the agent, or the harness: an ACQUIRE with a judge-built query, an
answer-surface re-acquire round, a recovery read), executes through the environment's call_tool, the single execution
point ActionExecutor.execute_and_normalize calls for agent and harness actions alike. The environment passes the current
task id and asks this guard twice per search:

    refuse_query(bench, query, task_id)      before the search runs. A query that explicitly targets the benchmark
                                             source (its name, dataset owner, paper title or arXiv id), or that
                                             literally contains the current item's provenance id (its PMCID, PMID or
                                             DOI), is refused: the backend is never called and the agent sees REFUSAL.
    filter_results(bench, entries, task_id)  after it. Every result whose link, title or snippet points at the benchmark
                                             source, or at the current item's direct material provenance, is removed;
                                             the agent sees what is left plus WITHHELD_NOTE.

Two kinds of rule, logged separately:

    benchmark_dataset_source      the section's refuse_query / urls / text rules: the dataset pages, the benchmark's
                                  paper, project page and code repository, their mirrors, copies of dataset rows.
    benchmark_provenance_source   the section's per-item table (`items`, keyed by task id): the article holding the
                                  item's original image or figure, that figure or caption page, and the article's
                                  supplement. An item's ids become URL, text and query rules (below); its title, byline,
                                  figure-caption and article-figure-page patterns are text rules. Other medical literature,
                                  including pages that cite the article, stays searchable.

A result that matches both kinds is counted once, under benchmark_dataset_source.

Only the current item's provenance applies. Another item's source article is not where this item was taken from, so on
this task it stays searchable. Without a task id (a probe outside any task) only the benchmark-level rules apply; a task
id missing from a non-empty item table raises, so the search fails visibly instead of running without its item rules.

Both calls return the call's blocked record, {"count": n, "patterns": [rule ids], "what": "query" | "results",
"reasons": {reason: n}}, or None. The environment puts it on the env call result ("blocked");
ActionExecutor.execute_and_normalize takes it off that result, so the agent's state, the observation string, the harness
and its ledger never receive it, and build_event puts it on the tool_call event; run.py sums it per task into
result.json "search_blocked" (task_counts), with "by_origin" and "by_reason".

The item table is read here and nowhere else. Nothing of it reaches the agent, the harness context, the ledger or an
observation: the agent-visible REFUSAL and WITHHELD_NOTE are the same text whatever the reason, and the blocked record,
which carries only counts, generic rule ids (item_pmcid, item_title, ...) and reasons, is kept on the tool_call event
and in result.json.

The rules are data, runner/search_blocklist.yaml, one section per benchmark. The filtering itself is hard-wired: there
is no switch that turns it off, and a missing or malformed blocklist raises (the search fails visibly) rather than
letting results through. Nothing here calls the network.
"""
import os
import re
import urllib.parse

import yaml

BLOCKLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_blocklist.yaml")

REASON_DATASET = "benchmark_dataset_source"
REASON_PROVENANCE = "benchmark_provenance_source"

# What the agent sees, the same text for both reasons. Neither names a rule or a reason, and neither starts with an error
# marker: a refused query or a withheld result is not a tool failure. REFUSAL and a fully withheld result list both say
# "no results", which the MedCTA plugin reads as a search that returned nothing usable.
REFUSAL = ("[search refused] this query targets the benchmark's own source (its dataset or paper), which search may "
           "not reach; no results returned.")
WITHHELD_NOTE = ("[search guard] %d result(s) withheld: they point at the benchmark's own source (its dataset, paper "
                 "or a copy of its rows).")

_RULE_LISTS = ("refuse_query", "urls", "text")
_CACHE = {}

# An item's provenance ids, their required form, and the rule id each becomes. Rule ids are generic on purpose: a
# blocked record names the kind of match, never the item's source.
_ITEM_IDS = (("pmcid", re.compile(r"^PMC\d+$")), ("pmid", re.compile(r"^\d+$")), ("doi", re.compile(r"^10\.\d{4,9}/\S+$")))
_ITEM_TEXT_IDS = ("item_title", "item_byline", "item_figure_caption", "item_article_figure")
# A rule with `field: title` is anchored at the start of a result's title: the title a live result carries, and the title
# part of a rendered line ("- <title>: <snippet>"; recorded observations carry nothing else). A page that only cites the
# article in its snippet is therefore not matched. A leading "[PDF]"-style marker is allowed.
_TITLE_MARK = r"(?:\[(?:PDF|HTML|DOC|DOCX|PPT)\]\s*)?"
# The literature indexes that key a page by a bare PubMed id (pubmed.ncbi.nlm.nih.gov/<pmid>/, ncbi.nlm.nih.gov/pubmed/<pmid>,
# europepmc.org/article/MED/<pmid>). A bare number elsewhere is not read as a PubMed id.
PMID_HOSTS = ("ncbi.nlm.nih.gov", "europepmc.org", "ebi.ac.uk")


def _flags(rule):
    return re.I if "i" in str(rule.get("flags") or "") else 0


def _compile(section_name, section):
    if not isinstance(section, dict):
        raise ValueError("search blocklist section %r is not a mapping" % section_name)
    out = {}
    for key in _RULE_LISTS:
        rules = section.get(key)
        if not isinstance(rules, list):
            raise ValueError("search blocklist %s.%s must be a list" % (section_name, key))
        compiled = []
        for rule in rules:
            rid = rule.get("id") if isinstance(rule, dict) else None
            if not rid:
                raise ValueError("search blocklist %s.%s has a rule without an id" % (section_name, key))
            if key == "urls":
                hosts = rule.get("hosts")
                if not hosts or not (rule.get("path") or rule.get("query")):
                    raise ValueError("search blocklist url rule %s.%s needs hosts and a path or query" % (section_name, rid))
                compiled.append({"id": rid, "hosts": [str(h).lower() for h in hosts],
                                 "path": re.compile(rule["path"], _flags(rule)) if rule.get("path") else None,
                                 "query": re.compile(rule["query"], _flags(rule)) if rule.get("query") else None})
            else:
                if not rule.get("pattern"):
                    raise ValueError("search blocklist rule %s.%s has no pattern" % (section_name, rid))
                compiled.append({"id": rid, "pattern": re.compile(rule["pattern"], _flags(rule))})
        out[key] = compiled
    items = section.get("items")
    if not isinstance(items, dict):
        raise ValueError("search blocklist %s.items must be a mapping (empty when the benchmark has no item table)"
                         % section_name)
    out["items"] = {str(tid): _compile_item(section_name, str(tid), item) for tid, item in items.items()}
    return out


def _id_rules(key, value):
    """The query, url and text rules one provenance id becomes (hard-wired; the table only holds the id)."""
    rid = "item_" + key
    if key == "pmid":
        return {"refuse_query": [{"id": rid, "pattern": re.compile(r"(?<![0-9])%s(?![0-9])" % value)}],
                "urls": [{"id": rid, "hosts": list(PMID_HOSTS), "query": None,
                          "path": re.compile(r"(?:^|/)%s(?:[/.?#]|$)" % value)},
                         {"id": rid, "hosts": list(PMID_HOSTS), "path": None,
                          "query": re.compile(r"(?:^|&)[^=&]*=(?:pmid:?|med/)?%s(?:&|$)" % value, re.I)}],
                "text": [{"id": rid, "pattern": re.compile(r"\bPMID\W{0,3}%s(?![0-9])" % value, re.I)}]}
    # A PMCID or a DOI names the article on any host, in a link's path or query (the article page, its figure pages
    # /figure/<id>/, its supplement /bin/<file>, its PDF, Europe PMC and publisher copies) and in a snippet. A PMCID
    # followed by a digit, or a DOI followed by a letter or digit, is another id; a DOI followed by "." (a figure or
    # supplement DOI such as <doi>.g005) is this article's.
    tail = r"(?![0-9])" if key == "pmcid" else r"(?![0-9A-Za-z])"
    pat = re.compile(r"(?<![0-9A-Za-z])%s%s" % (re.escape(value), tail), re.I)
    return {"refuse_query": [{"id": rid, "pattern": pat}],
            "urls": [{"id": rid, "hosts": ["*"], "path": pat, "query": None},
                     {"id": rid, "hosts": ["*"], "path": None, "query": pat}],
            "text": [{"id": rid, "pattern": pat}]}


def _compile_item(section_name, tid, item):
    """One item's compiled provenance rules, or None for an item whose provenance is recorded unknown."""
    where = "search blocklist %s.items.%s" % (section_name, tid)
    if not isinstance(item, dict):
        raise ValueError("%s is not a mapping" % where)
    status = item.get("status")
    if status == "unknown":
        if not item.get("evidence"):
            raise ValueError("%s is unknown without the evidence that was checked" % where)
        return None
    if status != "known":
        raise ValueError("%s.status must be known or unknown" % where)
    ids = item.get("ids")
    if not isinstance(ids, dict) or not any(ids.get(k) for k, _ in _ITEM_IDS):
        raise ValueError("%s is known without a PMCID, PMID or DOI" % where)
    out = {"refuse_query": [], "urls": [], "text": []}
    for key, form in _ITEM_IDS:
        value = ids.get(key)
        if value is None:
            continue
        value = str(value)
        if not form.match(value):
            raise ValueError("%s.ids.%s %r is not a %s" % (where, key, value, key))
        for lst, rules in _id_rules(key, value).items():
            out[lst].extend(rules)
    for rule in item.get("text") or []:
        rid = rule.get("id") if isinstance(rule, dict) else None
        if rid not in _ITEM_TEXT_IDS or not rule.get("pattern"):
            raise ValueError("%s.text rules need an id in %s and a pattern" % (where, ", ".join(_ITEM_TEXT_IDS)))
        field = rule.get("field") or "any"
        if field not in ("any", "title"):
            raise ValueError("%s.text rule %s: field must be title or any" % (where, rid))
        pattern = (_TITLE_MARK + "(?:%s)" % rule["pattern"]) if field == "title" else rule["pattern"]
        out["text"].append({"id": rid, "pattern": re.compile(pattern, _flags(rule)), "field": field,
                            "also": re.compile(rule["also"], _flags(rule)) if rule.get("also") else None})
    return out


def blocklist(path=BLOCKLIST_PATH):
    """{benchmark: {refuse_query: [...], urls: [...], text: [...], items: {task id: rules or None}}} compiled, cached
    per file."""
    if path not in _CACHE:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or not data:
            raise ValueError("search blocklist %s is empty or not a mapping" % path)
        _CACHE[path] = {name: _compile(name, sec) for name, sec in data.items()}
    return _CACHE[path]


def _section(bench):
    sec = blocklist().get(bench)
    if sec is None:
        raise KeyError("search blocklist has no section for benchmark %r" % bench)
    return sec


def _item(bench, task_id):
    """The current item's compiled provenance rules, or None (no task id, no item table, or provenance unknown)."""
    items = _section(bench)["items"]
    if task_id is None or not items:
        return None
    if str(task_id) not in items:
        raise KeyError("search blocklist %s has no item entry for task %r" % (bench, task_id))
    return items[str(task_id)]


def _record(ids, count, what, reasons):
    return {"count": count, "patterns": sorted(set(ids)), "what": what, "reasons": dict(sorted(reasons.items()))}


def refuse_query(bench, query, task_id=None):
    """The blocked record for a query that targets the benchmark source or names the current item's provenance id,
    else None."""
    q = str(query or "")
    item = _item(bench, task_id)
    ids = [r["id"] for r in _section(bench)["refuse_query"] if r["pattern"].search(q)]
    if ids:
        return _record(ids, 1, "query", {REASON_DATASET: 1})
    ids = [r["id"] for r in (item["refuse_query"] if item else []) if r["pattern"].search(q)]
    return _record(ids, 1, "query", {REASON_PROVENANCE: 1}) if ids else None


def _host_matches(host, hosts):
    return any(h == "*" or host == h or host.endswith("." + h) for h in hosts)


def _url_ids(rules, link):
    if not link:
        return []
    try:
        u = urllib.parse.urlsplit(str(link).strip())
    except ValueError:
        return []
    host = (u.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return []
    path, query = urllib.parse.unquote(u.path or ""), urllib.parse.unquote(u.query or "")
    ids = []
    for r in rules:
        if not _host_matches(host, r["hosts"]):
            continue
        if r["path"] is not None and not r["path"].search(path):
            continue
        if r["query"] is not None and not r["query"].search(query):
            continue
        ids.append(r["id"])
    return ids


def _titles(entry):
    """The title a live result carries, and the title part of its rendered line ("- <title>: <snippet>")."""
    out = [str(entry["title"]).strip()] if entry.get("title") else []
    text = str(entry.get("text") or "")
    if text.startswith("- "):
        out.append(text[2:])
    return out


def _text_ids(rules, texts, titles=()):
    blob = "\n".join(str(t) for t in texts if t)
    if not blob:
        return []
    ids = []
    for r in rules:
        if r.get("field") == "title":
            hit = any(r["pattern"].match(t) for t in titles)
        else:
            hit = bool(r["pattern"].search(blob))
        if hit and (r.get("also") is None or r["also"].search(blob)):
            ids.append(r["id"])
    return ids


def _entry_ids(rules, entry):
    links = entry.get("link")
    ids = []
    for link in (links if isinstance(links, (list, tuple)) else [links]):
        ids.extend(_url_ids(rules["urls"], link))
    return ids + _text_ids(rules["text"], (entry.get("title"), entry.get("snippet"), entry.get("text")), _titles(entry))


def url_matches(bench, link):
    """Rule ids of the benchmark-level url rules a result link matches."""
    return _url_ids(_section(bench)["urls"], link)


def text_matches(bench, *texts):
    """Rule ids of the benchmark-level text rules any of the given strings matches."""
    return _text_ids(_section(bench)["text"], texts)


def entry_hits(bench, entry, task_id=None):
    """(benchmark-level rule ids, current item's provenance rule ids) one result entry matches. An entry is
    {"text": rendered line, optional "link" (one or a list), "title", "snippet"}."""
    item = _item(bench, task_id)
    return _entry_ids(_section(bench), entry), (_entry_ids(item, entry) if item else [])


def entry_matches(bench, entry, task_id=None):
    """Every rule id one result entry matches (benchmark level, then the current item's)."""
    dataset, provenance = entry_hits(bench, entry, task_id)
    return dataset + provenance


def filter_results(bench, entries, task_id=None):
    """(kept entries, blocked record or None). Order of the kept entries is preserved."""
    item = _item(bench, task_id)
    sec = _section(bench)
    kept, ids, n, reasons = [], [], 0, {}
    for e in entries:
        dataset = _entry_ids(sec, e)
        provenance = _entry_ids(item, e) if item else []
        if dataset or provenance:
            n += 1
            ids.extend(dataset + provenance)
            reason = REASON_DATASET if dataset else REASON_PROVENANCE
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            kept.append(e)
    return kept, (_record(ids, n, "results", reasons) if n else None)


def task_counts(trajectory):
    """Per-task totals of the blocked records on tool_call events, split by origin and by reason (result.json
    "search_blocked"). A call is counted under every reason it carries."""
    total = {"calls": 0, "results_removed": 0, "queries_refused": 0}
    by_origin, by_reason = {}, {}
    for ev in trajectory or []:
        b = ev.get("blocked") if isinstance(ev, dict) and ev.get("event_type") == "tool_call" else None
        if not isinstance(b, dict):
            continue
        key = "queries_refused" if b.get("what") == "query" else "results_removed"
        o = by_origin.setdefault(ev.get("origin") or "agent", {"calls": 0, "results_removed": 0, "queries_refused": 0})
        for d in (total, o):
            d["calls"] += 1
            d[key] += int(b.get("count") or 0)
        for reason, n in sorted((b.get("reasons") or {}).items()):
            r = by_reason.setdefault(reason, {"calls": 0, "results_removed": 0, "queries_refused": 0})
            r["calls"] += 1
            r[key] += int(n or 0)
    total["by_origin"] = by_origin
    total["by_reason"] = by_reason
    return total
