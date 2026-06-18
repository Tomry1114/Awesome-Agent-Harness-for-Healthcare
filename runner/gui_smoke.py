import json
from playwright.sync_api import sync_playwright
url = "http://localhost:3002/emr/denied"
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--no-sandbox"])
    pg = b.new_page()
    pg.goto(url, wait_until="domcontentloaded", timeout=45000)
    pg.wait_for_timeout(2000)
    print("TITLE:", pg.title())
    print("URL:", pg.url)
    btns = pg.evaluate("() => Array.from(document.querySelectorAll('button,a,[role=button]')).slice(0,20).map(e=>(e.innerText||'').trim()).filter(Boolean)")
    print("CLICKABLE:", json.dumps(btns, ensure_ascii=False)[:600])
    st = pg.evaluate("() => { try { return localStorage.getItem('portals_state'); } catch(e){ return 'ERR:'+e; } }")
    print("LOCALSTORAGE portals_state:", (st[:400] if st else st))
    b.close()
print("SMOKE OK")
