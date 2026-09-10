"""
Build a self-contained HTML gallery of every pooled Gemini Likert Select image
scoring >=40, grouped by how much is known about it.

Tiers are mutually exclusive and ordered by increasing "knownness":
  unknown     no catalogue counterpart within 3", not flagged by AnomalyMatch
  amonly      flagged by AnomalyMatch, still no catalogue counterpart
  catalogued  a catalogue records the position, but no paper discusses it
              individually (split in the page by AnomalyMatch flag)
  discussed   at least one paper discusses the object individually, per the
              ADS full-text + snippet classification

Images are embedded as lossless PNG data URIs so the file works offline and
faint morphology is not smeared by lossy re-encoding.
"""

import os, io, json, base64, html

# The stage folders are siblings, so put the analysis root on the path to
# reach common/paths.py (see its docstring).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import TIER_REPORT_HTML, CACHE_DIR

# Tier assignments and the pre-encoded PNG payload are produced ad hoc (see
# the module docstring); point SCRATCH at wherever they were written.
SCRATCH = os.environ.get("TIER_SCRATCH_DIR", os.path.join(CACHE_DIR, "tier_inputs"))
OUT = TIER_REPORT_HTML

rows = json.load(open(os.path.join(SCRATCH, "tier_rows.json")))
imgs = json.load(open(os.path.join(SCRATCH, "imgs_png_b64.json")))

TIERS = [
    ("unknown",    "Unidentified",
     "No catalogue counterpart within 3″ and not flagged by AnomalyMatch. "
     "These are the candidates reported in the paper."),
    ("amonly",     "AnomalyMatch only",
     "Flagged by AnomalyMatch, but still no catalogue counterpart within 3″."),
    ("catalogued", "Catalogued, not discussed",
     "A catalogue records an object at this position, but no paper discusses it "
     "individually. This is the 313."),
    ("discussed",  "Discussed in the literature",
     "At least one paper treats the object individually, judged from verbatim "
     "ADS in-body snippets."),
]

for r in rows:
    r["img"] = imgs[r["fn"]]
rows.sort(key=lambda r: (-r["score"], r["fn"]))

n = {k: sum(1 for r in rows if r["tier"] == k) for k, _, _ in TIERS}
cat_am = sum(1 for r in rows if r["tier"] == "catalogued" and r["am"])
cat_un = sum(1 for r in rows if r["tier"] == "catalogued" and not r["am"])


def card(r):
    e = html.escape
    badges = []
    if r["am"]:
        t = f" &middot; {e(r['amtype'])}" if r["amtype"] else ""
        badges.append(f'<span class="b am">AnomalyMatch{t}</span>')
    if r["matched"]:
        badges.append(f'<span class="b cat">{e(r["src"] or "catalogue")}</span>')
    if r["disc"]:
        badges.append('<span class="b disc">discussed</span>')

    meta = [f'<div class="coord"><span>{r["ra"]:.6f}</span><span>{r["dec"]:+.6f}</span></div>']
    if r["obj"]:
        ot = f' <em>{e(r["otype"])}</em>' if r["otype"] else ""
        meta.append(f'<div class="obj">{e(r["obj"])}{ot}</div>')
    if r["npapers"] not in ("", None):
        meta.append(f'<div class="pap">{e(str(r["npapers"]))} papers checked</div>')
    if r["reason"]:
        meta.append(f'<details><summary>verdict</summary><p>{e(r["reason"])}</p></details>')

    q = f'{r["ra"]:.6f}%20{r["dec"]:+.6f}'
    links = (
        f'<a href="https://simbad.cds.unistra.fr/simbad/sim-coo?Coord={q}&Radius=10&Radius.unit=arcsec"'
        f' target="_blank" rel="noopener">SIMBAD</a>'
        f'<a href="https://ned.ipac.caltech.edu/conesearch?search_type=Near%20Position%20Search'
        f'&in_csys=Equatorial&in_equinox=J2000&ra={r["ra"]:.6f}&dec={r["dec"]:+.6f}&radius=0.17"'
        f' target="_blank" rel="noopener">NED</a>'
        f'<a href="https://sky.esa.int/esasky/?target={r["ra"]:.6f}%20{r["dec"]:+.6f}'
        f'&hips=DSS2+color&fov=0.05" target="_blank" rel="noopener">ESASky</a>'
    )
    return (
        f'<figure class="card" data-tier="{r["tier"]}" data-am="{r["am"]}" '
        f'data-score="{r["score"]}" data-id="{e(r["fn"])}" data-obj="{e(r["obj"].lower())}">'
        f'<img src="data:image/png;base64,{r["img"]}" alt="cutout {e(r["fn"])}" loading="lazy">'
        f'<figcaption>'
        f'<div class="top"><span class="score">{r["score"]:g}<small>/50</small></span>'
        f'<code>{e(r["fn"])}</code></div>'
        f'<div class="badges">{"".join(badges)}</div>'
        f'{"".join(meta)}'
        f'<div class="links">{links}</div>'
        f'</figcaption></figure>'
    )


sections = []
for key, title, blurb in TIERS:
    sub = [r for r in rows if r["tier"] == key]
    if key == "catalogued":
        groups = [("AnomalyMatch-flagged", [r for r in sub if r["am"]]),
                  ("Not flagged by AnomalyMatch", [r for r in sub if not r["am"]])]
    else:
        groups = [(None, sub)]
    body = []
    for gname, g in groups:
        if gname:
            body.append(f'<h3>{gname} <span class="c">{len(g)}</span></h3>')
        body.append('<div class="grid">' + "".join(card(r) for r in g) + "</div>")
    sections.append(
        f'<section id="{key}"><h2>{title} <span class="c">{len(sub)}</span></h2>'
        f'<p class="blurb">{blurb}</p>{"".join(body)}</section>'
    )

CSS = """
:root{
  --bg:#0e1116; --panel:#161b22; --line:#2a323d; --ink:#e6edf3; --dim:#8b949e;
  --accent:#58a6ff; --am:#d29922; --cat:#8957e5; --disc:#3fb950; --new:#f85149;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{padding:28px 24px 18px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0 0 6px;font-size:20px;letter-spacing:-.01em}
.sub{color:var(--dim);margin:0 0 18px;max-width:70ch}
.funnel{display:flex;flex-wrap:wrap;gap:10px}
.stat{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:132px}
.stat b{display:block;font-size:20px;font-variant-numeric:tabular-nums}
.stat span{color:var(--dim);font-size:12px}
.controls{position:sticky;top:0;z-index:9;display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  padding:12px 24px;background:rgba(22,27,34,.96);border-bottom:1px solid var(--line);
  backdrop-filter:blur(6px)}
button,select,input{font:inherit;color:var(--ink);background:var(--bg);
  border:1px solid var(--line);border-radius:6px;padding:6px 10px}
button{cursor:pointer}
button.on{border-color:var(--accent);color:var(--accent)}
input[type=search]{min-width:200px}
.controls .sp{flex:1}
main{padding:8px 24px 60px}
section{margin:34px 0 0}
h2{font-size:17px;margin:0 0 4px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin:22px 0 8px;font-weight:600}
.c{display:inline-block;background:var(--line);color:var(--ink);border-radius:20px;
  padding:1px 9px;font-size:12px;vertical-align:2px;font-variant-numeric:tabular-nums}
.blurb{color:var(--dim);margin:6px 0 14px;max-width:80ch}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(178px,1fr))}
.card{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card img{display:block;width:100%;aspect-ratio:1;object-fit:cover;background:#000;cursor:zoom-in}
figcaption{padding:8px 9px 9px;font-size:12px}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:6px}
.score{font-weight:700;font-variant-numeric:tabular-nums}
.score small{font-weight:400;color:var(--dim)}
.top code{color:var(--dim);font-size:10.5px}
.badges{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}
.b{font-size:10px;border-radius:4px;padding:1px 5px;border:1px solid}
.b.am{color:var(--am);border-color:var(--am)}
.b.cat{color:var(--cat);border-color:var(--cat)}
.b.disc{color:var(--disc);border-color:var(--disc)}
.coord{display:flex;justify-content:space-between;color:var(--dim);
  font-variant-numeric:tabular-nums;font-size:11px}
.obj{margin-top:4px;word-break:break-word}
.obj em{color:var(--dim);font-style:normal}
.pap{color:var(--dim);font-size:11px;margin-top:2px}
details{margin-top:5px}
summary{cursor:pointer;color:var(--accent);font-size:11px}
details p{margin:5px 0 0;color:var(--dim);font-size:11px}
.links{display:flex;gap:8px;margin-top:7px;padding-top:6px;border-top:1px solid var(--line)}
.links a{color:var(--dim);text-decoration:none;font-size:11px}
.links a:hover{color:var(--accent);text-decoration:underline}
.jump{color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px;
  padding:6px 10px;text-decoration:none;font-size:13px}
.jump:hover{border-color:var(--accent);color:var(--accent)}
.hide{display:none!important}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;place-items:center;z-index:99;cursor:zoom-out}
#lb.show{display:grid}
#lb img{width:min(78vmin,620px);height:auto;image-rendering:pixelated;
  border:1px solid var(--line);border-radius:6px}
#lb figure{margin:0}
#lb figcaption{color:var(--dim);text-align:center;margin-top:10px;font-size:12px;padding:0}
.empty{color:var(--dim);padding:10px 0}
"""

JS = """
const cards=[...document.querySelectorAll('.card')];
const amSel=document.getElementById('am');
const q=document.getElementById('q');
const sort=document.getElementById('sort');
function apply(){
  const a=amSel.value, t=q.value.trim().toLowerCase();
  cards.forEach(c=>{
    let ok=true;
    if(a==='yes'&&c.dataset.am!=='1')ok=false;
    if(a==='no'&&c.dataset.am!=='0')ok=false;
    if(t&&!(c.dataset.id.includes(t)||c.dataset.obj.includes(t)))ok=false;
    c.classList.toggle('hide',!ok);
  });
  document.querySelectorAll('.grid').forEach(g=>{
    const vis=[...g.children].filter(c=>!c.classList.contains('hide')).length;
    let e=g.nextElementSibling;
    if(e&&e.classList.contains('empty'))e.remove();
    if(!vis)g.insertAdjacentHTML('afterend','<p class="empty">nothing matches in this group</p>');
  });
}
function resort(){
  const dir=sort.value==='asc'?1:-1;
  document.querySelectorAll('.grid').forEach(g=>{
    [...g.children]
      .sort((x,y)=>dir*(x.dataset.score-y.dataset.score)||x.dataset.id.localeCompare(y.dataset.id))
      .forEach(c=>g.appendChild(c));
  });
}
amSel.onchange=apply; q.oninput=apply; sort.onchange=resort;
const lb=document.getElementById('lb'), lbi=lb.querySelector('img'),
      lbc=lb.querySelector('figcaption');
document.addEventListener('click',e=>{
  if(e.target.tagName==='IMG'&&e.target.closest('.card')){
    const c=e.target.closest('.card');
    lbi.src=e.target.src;
    lbc.textContent=c.dataset.id+'  \\u00b7  '+c.dataset.score+'/50';
    lb.classList.add('show');
  } else if(e.target.closest('#lb')) lb.classList.remove('show');
});
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.classList.remove('show')});
"""

doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Score &ge;40 cutouts by what is known about them</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Gemini <i>Likert Select</i> images scoring &ge;40, by what is known about them</h1>
  <p class="sub">All {len(rows)} images from the pooled three-seed ranking that score at least
  40/50, grouped into mutually exclusive tiers of increasing &ldquo;knownness&rdquo;. Catalogue
  matching is a staged 3&Prime; positional search (galaxy-mentions, SIMBAD, NED); the
  discussed/not-discussed verdict comes from the ADS full-text search with verbatim in-body
  snippets. Click any cutout to enlarge.</p>
  <div class="funnel">
    <div class="stat"><b>{len(rows)}</b><span>images &ge;40</span></div>
    <div class="stat"><b>{n['unknown']}</b><span>unidentified</span></div>
    <div class="stat"><b>{n['amonly']}</b><span>AnomalyMatch only</span></div>
    <div class="stat"><b>{n['catalogued']}</b><span>catalogued, not discussed</span></div>
    <div class="stat"><b>{cat_am} / {cat_un}</b><span>&hellip; AM-flagged / not</span></div>
    <div class="stat"><b>{n['discussed']}</b><span>discussed in papers</span></div>
  </div>
</header>
<div class="controls">
  <label>AnomalyMatch
    <select id="am"><option value="all">any</option><option value="yes">flagged</option>
    <option value="no">not flagged</option></select></label>
  <label>score
    <select id="sort"><option value="desc">high &rarr; low</option>
    <option value="asc">low &rarr; high</option></select></label>
  <input id="q" type="search" placeholder="filter by SourceID or object name">
  <span class="sp"></span>
  <a class="jump" href="#unknown">unidentified</a>
  <a class="jump" href="#catalogued">the 313</a>
  <a class="jump" href="#discussed">discussed</a>
</div>
<main>{"".join(sections)}</main>
<div id="lb"><figure><img alt="enlarged cutout"><figcaption></figcaption></figure></div>
<script>{JS}</script>
</body></html>"""

io.open(OUT, "w", encoding="utf-8").write(doc)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")
print(f"tiers: {n}, catalogued split {cat_am} AM-flagged / {cat_un} not")
