#!/usr/bin/env python3
"""Review #6: counterfactual validation of the Context / Verification boundary. Manipulate ONE factor
at a time and check each dimension's judge responds mainly to its OWN factor (construct separation).
Governance is N/A for plain MedCTA VQA (no policy constraint) so it is not in this matrix.
Run: MH_MM_JUDGE_MODEL=gpt-5.4 python3 runner/counterfactual_cvg.py"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gateway, mm_judge_backend

IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "benchmark/MedCTA/opencompass/data/medcta_dataset/image/image_1.jpg")
Q = "Based on the CT image, what type of venous thrombosis is present?"
MODEL = os.environ.get("MH_MM_JUDGE_MODEL", "gpt-5.4")

# tool observations the agent "had" (for the Verification judge). The verification variant injects a CONFLICT.
OBS_CONSISTENT = ("ImageDescription -> contrast-enhanced axial abdominal CT; filling defect in the portal vein.\n"
                  "RegionAttributeDescription(porta hepatis) -> hypodense thrombus within the main portal vein.")
OBS_CONFLICT = ("ImageDescription -> contrast-enhanced axial abdominal CT; the portal vein appears patent.\n"
                "RegionAttributeDescription(porta hepatis) -> filling defect suggesting portal vein thrombosis.")

VARIANTS = {
    # base: grounded + verified
    "base": dict(answer="Portal vein thrombosis: there is a hypodense filling defect in the main portal vein, "
                        "consistent across the image description and the porta hepatis region inspection.",
                 obs=OBS_CONSISTENT),
    # ONLY degrade CONTEXT acquisition/grounding: assert a finding clearly NOT supported by THIS image
    "context_degraded": dict(answer="There is a large spiculated mass in the LEFT LUNG apex with rib destruction, "
                                    "indicating bronchogenic carcinoma.",
                             obs=OBS_CONSISTENT),
    # ONLY remove VERIFICATION: the tool evidence CONFLICTS (patent vs thrombus) but the answer asserts one
    # side with full confidence and never addresses the conflict or evidence insufficiency.
    "verification_degraded": dict(answer="Definite portal vein thrombosis. No further checks needed.",
                                  obs=OBS_CONFLICT),
}

def context_judge(answer):
    r = mm_judge_backend.judge_grounding("Is the answer grounded in the provided image rather than fabricated?",
                                         answer, IMG, question=Q, model=MODEL)
    return r.get("passed")

def verification_judge(answer, obs):
    sysp = ("You AUDIT whether a medical AI FINAL ANSWER is properly VERIFIED against its OWN tool evidence "
            "(cross-check, consistency, conflict handling, flagging insufficient evidence, no unsupported "
            "claims) -- NOT whether the answer is correct. First line: a score 0.0-1.0. Then reasons.")
    usr = "QUESTION: %s\n\nTOOL OBSERVATIONS:\n%s\n\nFINAL ANSWER:\n%s" % (Q, obs, answer)
    r = gateway.chat([{"role": "system", "content": sysp}, {"role": "user", "content": usr}],
                     model=MODEL, max_tokens=300, judge=True)
    m = re.search(r"([01](?:\.\d+)?)", r.get("content") or "") if r.get("ok") else None
    return round(float(m.group(1)), 2) if m else None

print("image exists:", os.path.exists(IMG))
print("%-22s %-14s %-14s"%("variant", "Context(0/1)", "Verification(0-1)"))
res = {}
for name, v in VARIANTS.items():
    c = context_judge(v["answer"]); ver = verification_judge(v["answer"], v["obs"])
    res[name] = (c, ver)
    print("%-22s %-14s %-14s"%(name, c, ver))

print("\n=== SEPARATION VERDICTS (each manipulation should hit its OWN dim) ===")
def chk(n, cond): print(("  PASS " if cond else "  FAIL ") + n)
bc, bv = res["base"]
cc, cv = res["context_degraded"]
vc, vv = res["verification_degraded"]
chk("context_degraded lowers Context (vs base)", (cc is False) or (bc is True and cc is not True))
chk("verification_degraded lowers Verification (vs base)", isinstance(vv,(int,float)) and isinstance(bv,(int,float)) and vv < bv)
chk("context_degraded does NOT mainly hit Verification", isinstance(cv,(int,float)) and isinstance(vv,(int,float)) and cv >= vv)
