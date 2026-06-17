import json, urllib.request, collections
BASE="http://localhost:38080/fhir"
def get(url):
    req=urllib.request.Request(url, headers={"Accept":"application/fhir+json"})
    return json.load(urllib.request.urlopen(req, timeout=60))
url=BASE+"/Observation?category=laboratory&_elements=code,valueQuantity&_count=1000"
# per code: total n, numeric n
agg=collections.defaultdict(lambda:{"n":0,"num":0,"loinc":False})
total=0; numeric_total=0; pages=0
while url and pages<200:
    b=get(url); pages+=1
    for e in b.get("entry",[]):
        o=e["resource"]; total+=1
        codings=o.get("code",{}).get("coding",[]) or [{}]
        cd=None
        for c in codings:
            if "loinc" in (c.get("system") or "").lower(): cd=c; break
        is_loinc = cd is not None
        if cd is None: cd=codings[0]
        key=cd.get("code") or "?"
        a=agg[key]; a["n"]+=1; a["loinc"]=is_loinc
        vq=o.get("valueQuantity")
        if vq and isinstance(vq.get("value"),(int,float)):
            a["num"]+=1; numeric_total+=1
    nxt=[l["url"] for l in b.get("link",[]) if l.get("relation")=="next"]
    url=nxt[0] if nxt else None
print("TOTAL_ALL", total, "TOTAL_NUMERIC", numeric_total)
# numeric-only coverage milestones: rank codes by numeric count (loinc only, numeric>0)
numrows=sorted([(k,a) for k,a in agg.items() if a["num"]>0 and a["loinc"]], key=lambda kv:-kv[1]["num"])
cum=0; ms={}
for i,(k,a) in enumerate(numrows,1):
    cum+=a["num"]
    for thr in (50,80,90,95):
        if thr not in ms and 100*cum/numeric_total>=thr: ms[thr]=i
print("NUMERIC_LOINC_CODES_TOTAL", len(numrows))
print("NUMERIC_COVERAGE_CODES_NEEDED", {str(k)+"%":v for k,v in sorted(ms.items())})
# sum of the 43 seed codes
seed=["2345-7","33914-3","2823-3","2160-0","4544-3","3094-0","17861-6","2951-2","718-7","2075-0","777-3","6690-2","2028-9","787-2","785-6","786-4","789-8","788-0","33037-3","1975-2","1742-6","1920-8","1751-7","2885-2","6768-6","10834-0","5905-5","770-8","736-9","714-6","706-2","2777-1","3016-3","2089-1","19123-9","2571-8","4548-4","2085-9","2093-3","2532-0","1968-7","2276-4","6301-6"]
ssum=sum(agg[c]["n"] for c in seed if c in agg)
snum=sum(agg[c]["num"] for c in seed if c in agg)
print("SEED43_COUNT_ALL", ssum, "pct_all=%.1f"%(100*ssum/total), "pct_numeric=%.1f"%(100*ssum/numeric_total))
print("SEED43_NUMERIC", snum, "pct_of_numeric=%.1f"%(100*snum/numeric_total))
