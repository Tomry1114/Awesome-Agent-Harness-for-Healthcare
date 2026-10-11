PORTAL OPERATING GUIDANCE (general -- applies to every admin task; describes HOW the portals work, NOT the answer):
- CREDENTIALS / ACCESS: You are already an authenticated provider user. Do NOT sign up, log in fresh, or
  invent credentials. Payer identifiers (member IDs), portal links, and fax numbers are read FROM the EMR:
  open the patient's Coverages tab (or the denial's remittance/coverage area) and record them there. Reach a
  payer portal through the in-EMR link -- "Start Appeal" on a denial, or the "Open Portal" button in the
  Coverages "Payer Portal Access" section (scroll down; it is below the coverage details). If you land on a
  bare login page with no way forward, you arrived without the EMR context: go back to the EMR and enter via
  that link instead of trying to authenticate.
- DOWNLOAD DOCUMENTS FIRST: Download every required document in the EMR BEFORE navigating to any payer or fax
  portal. To open a document, click its "View" control on the RIGHT of the row (not the document name), then
  use the download action on the viewer page; use back EXACTLY ONCE to return to the referral/denial.
  Downloaded files then appear automatically in the "Available Documents from EMR" section inside a portal,
  where you attach them (click "+ Attach" next to each). If a required document is missing from that section
  it was never downloaded -- return to the EMR and download it; scrolling will not make it appear. Never click
  "Choose Files": that opens an OS dialog you cannot operate -- use the download/attach actions.
- DROPDOWNS are custom widgets, not native <select>. Prefer select {ref, option}; if a value will not set,
  operate it in two steps: click the dropdown control to open the option list, then click the option for the
  value you want. Enter dates as text in MM/DD/YYYY.
- PRIOR-AUTH / SUBMISSION FORMS follow an ordered flow: provider -> request type -> patient (look up by member
  ID + DOB) -> each diagnosis code (add) -> servicing provider -> each CPT/HCPCS code (add) -> clinical
  indication -> attach documents -> submit; then capture the confirmation / authorization ID. Multi-step
  portals advance page by page (Next) -- do NOT use back mid-form or in-progress entries are lost.
- TRIAGE / DISPOSITION NOTES: once you begin filling a disposition or triage note, do NOT click any other tab
  or navigate away -- switching tabs clears the note field. Set the disposition, type the note, and submit in
  one pass.
- FINISH IN THE EMR: after a portal submission, use "Return to EMR" to go back, add a note that records the
  confirmation number, and clear the item from the worklist if the task asks for it. This guidance tells you
  HOW the portals operate; you must still determine the correct disposition, rationale, codes, and field
  values from the patient's record yourself.
