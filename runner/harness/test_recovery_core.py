import sys; sys.path.insert(0,"runner")
from harness.evidence_state import (classify_evidence_state, AcquisitionKey, AcquisitionLog,
                                    PRESENT, ABSENT, UNKNOWN, FAILED, is_resolved, is_decision_changing)
R=[]
def ck(n,c): R.append((n,bool(c))); print(("OK  " if c else "FAIL")+" "+n)

# evidence_state
sem={"collection_paths":["entries"],"absence_when_empty":True}
ck("es_present", classify_evidence_state({"entries":[{"x":1}]}, sem)==PRESENT)
ck("es_absent",  classify_evidence_state({"entries":[]}, sem)==ABSENT)
ck("es_failed",  classify_evidence_state({"error":"boom"}, sem)==FAILED)
ck("es_unknown", classify_evidence_state({"timeout":True}, sem)==UNKNOWN)
ck("es_absent_resolves", is_resolved(ABSENT) and not is_decision_changing(ABSENT))
ck("es_present_changes", is_decision_changing(PRESENT))

# dedup
log=AcquisitionLog(); k=AcquisitionKey("Patient/1","AllergyIntolerance","",0)
ck("dedup_first", log.should_acquire(k))
log.record(k, ABSENT)
ck("dedup_resolved_no_repeat", not log.should_acquire(k))   # ABSENT resolved -> never repeat (no repeated PB query)
k2=AcquisitionKey("Patient/1","MedicationRequest","",0); log.record(k2, UNKNOWN)
ck("dedup_unknown_retry_once", log.should_acquire(k2, max_retry=1))
log.record(k2, UNKNOWN)
ck("dedup_unknown_stop", not log.should_acquire(k2, max_retry=1))

ok=sum(1 for _,c in R if c); print("\n%d/%d passed"%(ok,len(R))); sys.exit(0 if ok==len(R) else 1)
