import sys; sys.path.insert(0,"runner")
from harness.evidence_state import (classify_evidence_state, AcquisitionKey, AcquisitionLog,
                                    PRESENT, ABSENT, UNKNOWN, FAILED, is_resolved, is_decision_changing)
from harness.slot import (CommitmentSlot, compute_slot_delta, missing_required_slots, adopt_slot_patch)
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
ck("dedup_resolved_no_repeat", not log.should_acquire(k))   # ABSENT resolved -> never repeat (fixes PB re-query)
k2=AcquisitionKey("Patient/1","MedicationRequest","",0); log.record(k2, UNKNOWN)
ck("dedup_unknown_retry_once", log.should_acquire(k2, max_retry=1))
log.record(k2, UNKNOWN)
ck("dedup_unknown_stop", not log.should_acquire(k2, max_retry=1))

# slot delta
A=[CommitmentSlot("diagnosis",value="pneumonia"), CommitmentSlot("meds",value="amoxicillin"), CommitmentSlot("followup",value="",required_by_goal=True)]
B=[CommitmentSlot("diagnosis",value="pneumonia"), CommitmentSlot("meds",value="azithromycin"), CommitmentSlot("followup",value="2 weeks",required_by_goal=True)]
dl=compute_slot_delta(A,B); dd={d.slot_id:d.change for d in dl}
ck("delta_unchanged", dd["diagnosis"]=="unchanged")
ck("delta_changed", dd["meds"]=="changed")
ck("delta_followup_changed", dd["followup"]=="changed")

# missing required
ck("missing_required", missing_required_slots(["diagnosis","followup","referral"], A)==["followup","referral"])

# slot-level promotion
adopt,reason=adopt_slot_patch(dl, supported_slot_ids={"meds","followup"})
ck("adopt_all_supported", adopt)
adopt2,r2=adopt_slot_patch(dl, supported_slot_ids={"meds"})  # followup change unsupported
ck("reject_unsupported", not adopt2 and "unsupported" in r2)
# required slot removed -> reject
C=[CommitmentSlot("followup",value="",required_by_goal=True)]
dl2=compute_slot_delta([CommitmentSlot("followup",value="2wk",required_by_goal=True)], C)
adopt3,r3=adopt_slot_patch(dl2, supported_slot_ids={"followup"})
ck("reject_required_removed", not adopt3 and "removed" in r3)

ok=sum(1 for _,c in R if c); print("\n%d/%d passed"%(ok,len(R))); sys.exit(0 if ok==len(R) else 1)
