import json, urllib.request, statistics, collections
BASE="http://localhost:38080/fhir"
def get(url):
    req=urllib.request.Request(url, headers={"Accept":"application/fhir+json"})
    return json.load(urllib.request.urlopen(req, timeout=60))

url=BASE+"/Observation?category=laboratory&_elements=code,valueQuantity&_count=1000"
agg=collections.defaultdict(lambda:{"n":0,"disp":collections.Counter(),"unit":collections.Counter(),"vals":[],"sys":""})
total=0; pages=0; nonnum=0
while url and pages<200:
    b=get(url); pages+=1
    for e in b.get("entry",[]):
        o=e["resource"]; total+=1
        codings=o.get("code",{}).get("coding",[]) or [{}]
        cd=None
        for c in codings:
            if "loinc" in (c.get("system") or "").lower(): cd=c; break
        if cd is None: cd=codings[0]
        sysn=(cd.get("system") or "").split("/")[-1].split(":")[-1]
        code=cd.get("code") or "?"
        key=(sysn,code)
        a=agg[key]; a["n"]+=1; a["sys"]=sysn
        disp=cd.get("display") or o.get("code",{}).get("text")
        if disp: a["disp"][disp]+=1
        vq=o.get("valueQuantity")
        if vq and isinstance(vq.get("value"),(int,float)):
            a["vals"].append(vq["value"]); 
            if vq.get("unit"): a["unit"][vq["unit"]]+=1
        else: nonnum+=1
    nxt=[l["url"] for l in b.get("link",[]) if l.get("relation")=="next"]
    url=nxt[0] if nxt else None

def pct(v,p):
    if not v: return None
    s=sorted(v); i=min(len(s)-1,int(round(p/100*(len(s)-1)))); return s[i]

rows=sorted(agg.items(), key=lambda kv:-kv[1]["n"])
distinct=len(rows)
print("TOTAL_LAB_OBS",total)
print("DISTINCT_CODES",distinct)
print("PAGES",pages,"NONNUMERIC",nonnum)
cum=0
print("RANK|SYSTEM|CODE|N|CUM%|UNIT|P2.5|P50|P97.5|DISPLAY")
for i,(k,a) in enumerate(rows[:80],1):
    cum+=a["n"]; 
    unit=a["unit"].most_common(1)[0][0] if a["unit"] else ""
    disp=a["disp"].most_common(1)[0][0] if a["disp"] else ""
    v=a["vals"]
    p_lo=pct(v,2.5); p_med=pct(v,50); p_hi=pct(v,97.5)
    def f(x): return ("%.3g"%x) if isinstance(x,(int,float)) else ""
    print("%d|%s|%s|%d|%.1f|%s|%s|%s|%s|%s"%(i,a["sys"],k[1],a["n"],100*cum/total,unit,f(p_lo),f(p_med),f(p_hi),disp[:48]))
# coverage milestones
cum=0; milestones={}
for i,(k,a) in enumerate(rows,1):
    cum+=a["n"]
    for thr in (50,80,90,95,99):
        if thr not in milestones and 100*cum/total>=thr: milestones[thr]=i
print("COVERAGE_CODES_NEEDED", {str(k)+"%":v for k,v in sorted(milestones.items())})
