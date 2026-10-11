ACTION SYNTAX (DOM / testid mode):
- Click: click([testid]) | Fill: fill([testid], "value") | Navigate: navigate_to("url") | Back: back()
- Dropdowns are custom (NOT native <select>). Two steps required:
    1. click([dropdown-testid]) to open the options list.
    2. click([dropdown-testid-option-{value}]) to select the desired option.
  Example: click([request-type-select]) then click([request-type-select-option-outpatient])
  Do NOT use select() — it will not work.
- Dates: fill the text field with MM/DD/YYYY (e.g. 03/15/1965).
