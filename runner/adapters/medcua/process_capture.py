"""Read-only private telemetry for the record_vitals evaluator; never an observation.

No expected values, native rewards or checker logic are read. This is adapter
instrumentation, not a patch to the vendored application.
"""
VERSION = 'medcua-vitals-state-2'
STATE_JS = r'''() => {
 const s = window._taskState;
 if (!s || !Array.isArray(s.patients)) return null;
 const save=document.querySelector('.vp button[onclick^="saveV("]');
 const match=save && /^saveV\((\d+)\)$/.exec(save.getAttribute('onclick'));
 const selected=match ? s.patients[Number(match[1])] : null;
 const visible = el => {
   if (!el) return false;
   const r=el.getBoundingClientRect(), style=getComputedStyle(el);
   return r.width>0 && r.height>0 && r.bottom>0 && r.top<innerHeight &&
          r.right>0 && r.left<innerWidth && style.visibility!=='hidden';
 };
 const panel=document.querySelector('.vp');
 const fields=['sbp','dbp','tempf','hr','rr','spo2','gcs'];
 const draft={};
 for (const f of fields) {
   const el=document.getElementById('v-'+f);
   draft[f]=el && el.value!=='' ? Number(el.value) : null;
 }
 const section=Array.from(document.querySelectorAll('.vp .vs')).find(
   el => el.querySelector('.vs-t')?.textContent.trim().replace(/[▶►]/g,'').trim()==='Vitals');
 const values=section ? Array.from(section.querySelectorAll('.vv')) : [];
 if (visible(panel) && !selected) throw Error('Patient binding unavailable');
 return {logged_in:!!s.loggedIn,
   selected_patient:visible(panel) && selected ? selected.id : null,
   draft:visible(panel) ? draft : null,
   recorded_visible:values.length>0 && values.every(visible),
   patients:s.patients.map(p => ({id:p.id, complaint:p.complaint,
     recorded:p.recordedVitals, pain:p.pain}))};
}'''
TARGET_JS = r'''([x,y]) => {
 const e=document.elementFromPoint(x,y);
 const b=e && e.closest('button');
 const match=b && /^saveV\((\d+)\)$/.exec(b.getAttribute('onclick') || '');
 const selected=match ? window._taskState?.patients[Number(match[1])] : null;
 return {control:b && b.textContent.trim()==='Save Vitals' ? 'save_vitals' : null,
         subject_id:selected ? selected.id : null};
}'''


def enabled(env_id):
    return env_id in ('medgym.emergency_triage.record_vitals.intent',
                      'medgym.emergency_triage.record_vitals.step')


def target(page, action):
    import ast
    call = ast.parse(action, mode='eval').body
    if getattr(call.func, 'id', None) != 'mouse_click':
        return None
    args = [ast.literal_eval(a) for a in call.args]
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    x, y = (args[:2] if len(args) >= 2 else (kw.get('x'), kw.get('y')))
    return page.evaluate(TARGET_JS, [x, y]) if x is not None and y is not None else None
