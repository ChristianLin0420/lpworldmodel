"""Self-contained interactive HTML report for the LpWM / Pi-WM campaign.

The offline PNG suite (analysis/figures.py) is the archival record; this is the
thing you open to ASK something. A PNG cannot answer "what does this look like
without the k-WTA arms" or "what exactly is that point" -- those are filter and
hover questions, and they are the ones actually asked when reading a campaign.

Design comes from analysis/panels.py via `css_tokens()`, not from a second palette
restated here. That is deliberate: the reason the old suite showed one arm in two
different colours on one page was exactly such a duplication.

No CDN, no build step, no runtime dependency: the output is one file that opens
from a filesystem path, which is what makes it survivable on a cluster.

Usage:
    python analysis/report.py --campaign campaign.json --runs 'runs/outputs/*' \
        --out figures/report.html
    python analysis/report.py --selftest --out /tmp/report.html
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import figures as FG  # noqa: E402
from analysis import panels as P  # noqa: E402


def _num(x):
    """float(x) or None -- JSON has no NaN, and a NaN silently breaks JSON.parse."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _series(y, cap=400):
    """Finite values, decimated to `cap` points.

    A 200k-point history embedded verbatim would make a 40 MB HTML file that no
    browser will scroll smoothly. Decimation is by stride rather than by averaging
    so the points plotted are points that were actually logged.
    """
    a = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    if a.size > cap:
        a = a[:: int(np.ceil(a.size / cap))]
    return [round(float(v), 6) for v in a]


def build_payload(arms=None, gates=None, gate_values=None, runs=(), metric="train_reg_loss"):
    """Everything the page needs, as plain JSON-able types.

    Kept separate from the rendering so it can be unit-tested without parsing HTML,
    and so a caller with data from somewhere else can render the same page.
    """
    arms = arms or {}
    contrasts = []
    for members in FG.group_arms(arms).values():
        for a in members[1:]:
            if FG.paired_effect(arms, members[0], a)["n"]:
                contrasts.append([members[0], a])
    sds = [FG.paired_effect(arms, c, v)["sd"] for c, v in contrasts]
    sds = [x for x in sds if np.isfinite(x)]
    ns = [FG.paired_effect(arms, c, v)["n"] for c, v in contrasts]
    mde = _num(FG.mde(float(np.median(sds)), int(np.median(ns)))) if sds else None

    by_arm = FG.by_arm(list(runs)) if runs else {}
    rows = []
    for name in sorted(set(arms) | set(by_arm)):
        seeds = [_num(v) for v in arms.get(name, {}).values()]
        seeds = [v for v in seeds if v is not None]
        hist = []
        for r in by_arm.get(name, []):
            if metric in r["hist"]:
                hist = _series(r["hist"][metric])
                break
        rows.append({
            "name": name, "slot": P.arm_slot(name), "marker": P.arm_style(name)["marker"],
            "dash": P.ARM_SPEC.get(P.canon_arm(name), (0, 0, 0, False))[3],
            "family": P.FAMILY_LABEL.get(P.arm_family(name), "other"),
            "control": P.is_control(name),
            "ctrl": next((c for c, v in contrasts if v == name), None),
            "seeds": seeds, "mean": _num(np.mean(seeds)) if seeds else None,
            "hist": hist,
            "gate": _series(np.asarray(gate_values.get(name, []), float).ravel(), cap=1500)
                    if gate_values else [],
        })
    return {"arms": rows, "contrasts": contrasts, "mde": mde, "metric": metric,
            "tokens": P.css_tokens(),
            "gates": [{"name": g.get("name", ""), "observed": _num(g.get("observed")),
                       "lo": _num(g.get("lo")), "hi": _num(g.get("hi")),
                       "threshold": _num(g.get("threshold")),
                       "direction": g.get("direction", "above")}
                      for g in (gates or [])]}


def _css(tokens):
    """The palette as CSS custom properties, declared for both theme scopes.

    Both the media query and the explicit `data-theme` scope are declared, and the
    toggle wins over the OS setting in both directions -- a viewer who picked light
    on an OS-dark machine must get light.
    """
    def block(mode):
        return "\n".join(f"  --{k}: {v};" for k, v in sorted(tokens[mode].items()))
    # Light is the DEFAULT and the OS setting does not override it. This report is
    # read next to papers and printed figures, and an OS-dark machine silently
    # flipping it would put the reader on a surface the PNGs were never validated
    # against. Dark remains one click away and uses its own validated steps.
    return f""":root {{ color-scheme: light;
{block("light")}
}}
:root[data-theme="dark"] {{ color-scheme: dark;
{block("dark")}
}}"""


HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
__REFRESH__<title>PiWM / LpWM campaign report</title><style>
__CSS__
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
 font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:26px 24px 60px}
h1{font-size:19px;font-weight:600;margin:0 0 2px}
.sub{color:var(--ink-2);font-size:12.5px;margin:0 0 20px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:12px 0 16px;
 border-bottom:1px solid var(--line);margin-bottom:22px}
.bar .lbl{font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-3);
 margin-right:4px;align-self:center}
.fam{display:flex;flex-direction:column;gap:5px;margin-right:14px}
.fam .h{font-size:10.5px;letter-spacing:.03em;color:var(--ink-3);text-transform:uppercase}
.fam .row{display:flex;gap:6px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);
 background:var(--surface-2);border-radius:999px;padding:5px 11px;font-size:12.5px;
 cursor:pointer;user-select:none;font-family:inherit}
.chip .sw{width:9px;height:9px;border-radius:2px;background:currentColor;flex:0 0 auto}
.chip.off{opacity:.4}
.spacer{flex:1}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(184px,1fr));gap:12px;margin-bottom:26px}
.tile{border:1px solid var(--line);border-radius:10px;padding:13px 15px;background:var(--surface-2)}
.tile .k{font-size:11.5px;color:var(--ink-3);letter-spacing:.03em}
.tile .v{font-size:29px;font-weight:600;letter-spacing:-.02em;margin:3px 0 1px}
.tile .d{font-size:12px;color:var(--ink-2)}
.card{border:1px solid var(--line);border-radius:12px;padding:16px 18px 12px;
 margin-bottom:20px;background:var(--surface)}
.card h2{font-size:14.5px;font-weight:600;margin:0 0 2px}
.card p.cap{font-size:12.5px;color:var(--ink-2);margin:0 0 14px}
svg{display:block;width:100%;height:auto;overflow:visible}
.gr{stroke:var(--hair);stroke-width:1;fill:none}
text{fill:var(--ink-2);font-size:11px;font-family:inherit}
text.tick{fill:var(--ink-3);font-size:10.5px;font-variant-numeric:tabular-nums}
text.lab{fill:var(--ink);font-size:11.5px}
.tip{position:fixed;pointer-events:none;z-index:40;background:var(--surface);
 border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;
 box-shadow:0 6px 22px rgba(0,0,0,.16);opacity:0;transition:opacity .08s}
.tip .r{display:flex;gap:9px;justify-content:space-between;color:var(--ink-2)}
.tip .r span:last-child{color:var(--ink);font-variant-numeric:tabular-nums}
.tgl{border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2);
 border-radius:8px;padding:6px 11px;font-size:12.5px;cursor:pointer;font-family:inherit}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--hair)}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
th{color:var(--ink-3);font-weight:500;font-size:11.5px}
.hidden{display:none}
.foot{font-size:12px;color:var(--ink-3);margin-top:6px}
.sm{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.sm figure{margin:0}.sm figcaption{font-size:11.5px;margin:0 0 2px;font-weight:500}
.empty{color:var(--ink-3);font-size:12.5px;padding:14px 0}
</style></head><body>
<div class="wrap">
<h1>PiWM / LpWM campaign</h1>
<p class="sub">__SUB__ &middot; generated __STAMP____LIVE__</p>
<div class="bar" id="filters"><span class="lbl">Arms</span><span class="spacer"></span>
<button class="tgl" id="theme">Dark</button>
<button class="tgl" id="tableview">Table view</button></div>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>Paired effect vs matched control</h2>
<p class="cap">Each dot is one seed; the bar is the arm mean. Below, the bootstrap
distribution of the paired difference. A hollow dot sits inside the n=3 detection
floor &mdash; underpowered, not null.</p>
<div id="est"></div><div class="foot" id="estfoot"></div></div>
<div class="card"><h2>__METRIC__ &mdash; one facet per arm</h2>
<p class="cap">Shared axes, every other arm ghosted behind. Hover for the value at
that step.</p><div class="sm" id="small"></div></div>
<div class="card" id="ridgecard"><h2>Gate magnitude &mdash; full distribution</h2>
<p class="cap">A mean can sit on target with no mass there. Rules at sigmoid &asymp; 0.5
and r&middot;softmax = 1.0.</p><div id="ridge"></div></div>
<div class="card hidden" id="tbl"><h2>Table view</h2>
<p class="cap">The WCAG-clean twin of every chart above.</p><div id="tblbody"></div></div>
</div><div class="tip" id="tip"></div>
<script>
const D = __DATA__;
"""
JS = r"""
const T = document.getElementById('tip');
const S = (t,a={})=>{const e=document.createElementNS('http://www.w3.org/2000/svg',t);
  for(const k in a) e.setAttribute(k,a[k]); return e;};
const txt = (e,s)=>{e.textContent=s; return e;};
const col = a => 'var(--'+a.slot+')';
const f3 = (v,n=3)=> v==null||!isFinite(v) ? '--' : v.toFixed(n);
const row = (k,v)=>`<div class="r"><span>${k}</span><span>${v}</span></div>`;
let active = new Set(D.arms.map(a=>a.name));
const on = () => D.arms.filter(a=>active.has(a.name));

function tip(html, ev){ T.innerHTML=html; T.style.opacity=1;
  const r=T.getBoundingClientRect();
  let x=ev.clientX+14, y=ev.clientY-10;
  if(x+r.width>innerWidth-8) x=ev.clientX-r.width-14;
  if(y+r.height>innerHeight-8) y=innerHeight-r.height-8;
  T.style.left=x+'px'; T.style.top=y+'px'; }
const untip = ()=>{T.style.opacity=0;};

/* filter row -- one row above everything it scopes, never per-card */
const fb=document.getElementById('filters');
/* chips grouped by family: the filter row then reads as the experiment's design
   (baseline / sparse / gating / union) instead of one flat list of nine names */
const fams=[]; D.arms.forEach(a=>{ if(!fams.includes(a.family)) fams.push(a.family); });
fams.forEach(fam=>{
  const box=document.createElement('div'); box.className='fam';
  box.innerHTML=`<span class="h">${fam}</span>`;
  const row=document.createElement('div'); row.className='row'; box.appendChild(row);
  D.arms.filter(a=>a.family===fam).forEach(a=>{
    const b=document.createElement('button');
    b.className='chip'; b.setAttribute('aria-pressed','true'); b.style.color=col(a);
    b.title=a.name;
    b.innerHTML=`<i class="sw"></i><span style="color:var(--ink)">${a.name}</span>`;
    b.onclick=()=>{ const isOn=b.getAttribute('aria-pressed')==='true';
      b.setAttribute('aria-pressed', isOn?'false':'true'); b.classList.toggle('off', isOn);
      isOn?active.delete(a.name):active.add(a.name); draw(); };
    row.appendChild(b); });
  fb.insertBefore(box, fb.querySelector('.spacer')); });
document.getElementById('theme').onclick=e=>{
  const dark=document.documentElement.dataset.theme==='dark';
  document.documentElement.dataset.theme=dark?'light':'dark';
  e.target.textContent=dark?'Dark':'Light'; };
document.getElementById('tableview').onclick=e=>{
  const t=document.getElementById('tbl'); t.classList.toggle('hidden');
  e.target.textContent=t.classList.contains('hidden')?'Table view':'Hide table'; };

function boot(d){ const bs=[];
  for(let b=0;b<1500;b++){ let s=0; for(let k=0;k<d.length;k++) s+=d[(Math.random()*d.length)|0];
    bs.push(s/d.length); }
  bs.sort((p,q)=>p-q); return bs; }

function estimation(){
  const host=document.getElementById('est'); host.innerHTML='';
  const order=[]; const seen=new Set();
  D.contrasts.forEach(([c,v])=>[c,v].forEach(n=>{ if(!seen.has(n)&&active.has(n)){seen.add(n);order.push(n);} }));
  const A=order.map(n=>D.arms.find(a=>a.name===n)).filter(a=>a&&a.seeds.length);
  if(!A.length){ host.innerHTML='<p class="empty">No arm with seeds selected.</p>'; return; }
  const W=1180,H=440,ML=72,MR=26,MT=16,MB=62,SPLIT=228;
  const xs=i=>ML+(i+.5)*(W-ML-MR)/A.length;
  const av=A.flatMap(a=>a.seeds), y0=Math.min(...av)-.02, y1=Math.max(...av)+.02;
  const yA=v=>MT+(y1-v)/((y1-y0)||1)*(SPLIT-MT-24);
  const idx=Object.fromEntries(A.map((a,i)=>[a.name,i]));
  const diffs={};
  D.contrasts.forEach(([c,v])=>{ if(!(v in idx)) return;
    const C=D.arms.find(a=>a.name===c); if(!C||!C.seeds.length) return;
    const V=D.arms.find(a=>a.name===v), n=Math.min(C.seeds.length,V.seeds.length);
    const d=V.seeds.slice(0,n).map((s,i)=>s-C.seeds[i]);
    const bs=boot(d); diffs[v]={m:d.reduce((p,q)=>p+q,0)/n, lo:bs[37], hi:bs[1462], bs, ctrl:c}; });
  const dv=Object.values(diffs).flatMap(d=>[d.lo,d.hi]).concat([0]);
  const d0=Math.min(...dv)-.012, d1=Math.max(...dv)+.012;
  const yB=v=>SPLIT+40+(d1-v)/((d1-d0)||1)*(H-SPLIT-40-MB+26);
  const g=S('svg',{viewBox:`0 0 ${W} ${H}`,role:'img',
    'aria-label':'Paired effect versus matched control for each arm'});
  for(let i=0;i<=4;i++){ const v=y0+(y1-y0)*i/4;
    g.appendChild(S('line',{class:'gr',x1:ML,x2:W-MR,y1:yA(v),y2:yA(v)}));
    g.appendChild(txt(S('text',{class:'tick',x:ML-9,y:yA(v)+4,'text-anchor':'end'}),v.toFixed(2))); }
  g.appendChild(txt(S('text',{x:14,y:MT+70,transform:`rotate(-90 14 ${MT+70})`,'text-anchor':'middle'}),
    'CEM success rate'));
  D.contrasts.forEach(([c,v])=>{ if(!(c in idx)||!(v in idx)) return;
    const C=D.arms.find(a=>a.name===c), V=D.arms.find(a=>a.name===v);
    if(C.mean==null||V.mean==null) return;
    g.appendChild(S('line',{x1:xs(idx[c]),x2:xs(idx[v]),y1:yA(C.mean),y2:yA(V.mean),
      stroke:'var(--ghost)','stroke-width':1})); });
  A.forEach((a,i)=>{ const c=col(a), x=xs(i);
    g.appendChild(S('line',{x1:x-19,x2:x+19,y1:yA(a.mean),y2:yA(a.mean),
      stroke:c,'stroke-width':2.4,'stroke-linecap':'round'}));
    a.seeds.forEach((s,k)=>{ const dx=(k-(a.seeds.length-1)/2)*11;
      const sq = a.marker==='s'||a.marker==='D';
      const mk = sq ? S('rect',{x:x+dx-4.4,y:yA(s)-4.4,width:8.8,height:8.8,fill:c,
                    stroke:'var(--surface)','stroke-width':2,
                    transform:a.marker==='D'?`rotate(45 ${x+dx} ${yA(s)})`:''})
                    : S('circle',{cx:x+dx,cy:yA(s),r:4.6,fill:c,stroke:'var(--surface)','stroke-width':2});
      mk.style.cursor='crosshair';
      mk.onmousemove=e=>tip(`<b>${a.name}</b>${row('seed '+k,f3(s))}${row('arm mean',f3(a.mean))}`+
        `${row('family',a.family)}`,e);
      mk.onmouseleave=untip; g.appendChild(mk); }); });
  if(D.mde!=null) g.appendChild(S('rect',{x:ML,y:yB(D.mde),width:W-ML-MR,
    height:Math.max(yB(-D.mde)-yB(D.mde),1),fill:'var(--hair)'}));
  g.appendChild(S('line',{x1:ML,x2:W-MR,y1:yB(0),y2:yB(0),stroke:'var(--ink-3)','stroke-width':1}));
  for(let i=0;i<=4;i++){ const v=d0+(d1-d0)*i/4;
    g.appendChild(txt(S('text',{class:'tick',x:ML-9,y:yB(v)+4,'text-anchor':'end'}),
      (v>0?'+':'')+v.toFixed(2))); }
  g.appendChild(txt(S('text',{x:14,y:yB((d0+d1)/2),transform:`rotate(-90 14 ${yB((d0+d1)/2)})`,
    'text-anchor':'middle'}),'paired effect vs control'));
  let resolved=0;
  A.forEach((a,i)=>{ const d=diffs[a.name]; if(!d) return; const c=col(a), x=xs(i);
    const NB=34, lo=d.bs[0], hi=d.bs[d.bs.length-1], h=new Array(NB).fill(0);
    d.bs.forEach(v=>h[Math.min(NB-1,Math.floor((v-lo)/((hi-lo)||1)*NB))]++);
    const mx=Math.max(...h)||1; let p='';
    h.forEach((n,k)=>{ p+=(k?'L':'M')+(x+n/mx*26)+','+yB(lo+(k+.5)/NB*(hi-lo)); });
    g.appendChild(S('path',{d:p+`L${x},${yB(hi)}L${x},${yB(lo)}Z`,fill:c,'fill-opacity':.20,
      stroke:c,'stroke-width':1}));
    g.appendChild(S('line',{x1:x,x2:x,y1:yB(d.lo),y2:yB(d.hi),stroke:c,'stroke-width':2.6,
      'stroke-linecap':'round'}));
    const res = D.mde!=null && Math.abs(d.m)>=D.mde; resolved+=res?1:0;
    const dot=S('circle',{cx:x,cy:yB(d.m),r:6,fill:res?c:'var(--surface)',stroke:c,'stroke-width':2.2});
    dot.style.cursor='crosshair';
    dot.onmousemove=e=>tip(`<b>${a.name}</b> &minus; ${d.ctrl}${row('effect',(d.m>0?'+':'')+f3(d.m))}`+
      `${row('95% CI',f3(d.lo)+' … '+f3(d.hi))}`+
      `${row('vs MDE '+(D.mde==null?'--':'±'+f3(D.mde)), res?'resolved':'underpowered')}`,e);
    dot.onmouseleave=untip; g.appendChild(dot);
    g.appendChild(txt(S('text',{class:'lab',x:x-12,y:yB(d.m)+4,'text-anchor':'end',fill:c}),
      (d.m>0?'+':'')+f3(d.m))); });
  A.forEach((a,i)=>{ const parts=a.name.split('_');
    g.appendChild(txt(S('text',{class:'tick',x:xs(i),y:H-30,'text-anchor':'middle'}),parts[0]));
    g.appendChild(txt(S('text',{class:'lab',x:xs(i),y:H-16,'text-anchor':'middle',fill:col(a)}),
      parts.slice(1).join('_'))); });
  host.appendChild(g);
  document.getElementById('estfoot').textContent =
    `${D.mde==null?'no detection floor estimable':'grey band = ± MDE '+f3(D.mde)+' at n=3 seeds'}`
    + ` · squares/diamonds = flags-off controls · ${resolved} of ${Object.keys(diffs).length} contrasts resolved`;
}

function small(){
  const host=document.getElementById('small'); host.innerHTML='';
  const A=on().filter(a=>a.hist.length>1);
  if(!A.length){ host.innerHTML='<p class="empty">No selected arm carries '+D.metric+'.</p>'; return; }
  const all=D.arms.flatMap(a=>a.hist); const y0=Math.min(...all), y1=Math.max(...all);
  A.forEach(a=>{ const W=360,H=152,ML=48,MR=10,MT=10,MB=26, N=a.hist.length;
    const X=i=>ML+(N>1? i/(N-1):0)*(W-ML-MR), Y=v=>MT+(y1-v)/((y1-y0)||1)*(H-MT-MB);
    const f=document.createElement('figure');
    f.innerHTML=`<figcaption style="color:${col(a)}">${a.name}</figcaption>`;
    const g=S('svg',{viewBox:`0 0 ${W} ${H}`});
    for(let i=0;i<=3;i++){ const v=y0+(y1-y0)*i/3;
      g.appendChild(S('line',{class:'gr',x1:ML,x2:W-MR,y1:Y(v),y2:Y(v)}));
      g.appendChild(txt(S('text',{class:'tick',x:ML-7,y:Y(v)+4,'text-anchor':'end'}),
        v.toPrecision(3))); }
    D.arms.forEach(b=>{ if(b.name===a.name||b.hist.length<2) return;
      const M=b.hist.length;
      g.appendChild(S('path',{d:b.hist.map((v,i)=>(i?'L':'M')+(ML+i/(M-1)*(W-ML-MR))+','+Y(v)).join(''),
        fill:'none',stroke:'var(--ghost)','stroke-width':1.1})); });
    g.appendChild(S('path',{d:a.hist.map((v,i)=>(i?'L':'M')+X(i)+','+Y(v)).join(''),fill:'none',
      stroke:col(a),'stroke-width':2.1,'stroke-dasharray':a.dash?'5 3':'none','stroke-linejoin':'round'}));
    const cur=S('line',{x1:0,x2:0,y1:MT,y2:H-MB,stroke:'var(--ink-3)','stroke-width':1,opacity:0});
    const hit=S('rect',{x:ML,y:MT,width:W-ML-MR,height:H-MT-MB,fill:'transparent'});
    hit.style.cursor='crosshair';
    hit.onmousemove=e=>{ const r=g.getBoundingClientRect();
      const i=Math.max(0,Math.min(N-1,Math.round(((e.clientX-r.left)/r.width*W-ML)/(W-ML-MR)*(N-1))));
      cur.setAttribute('x1',X(i)); cur.setAttribute('x2',X(i)); cur.setAttribute('opacity',1);
      tip(`<b>${a.name}</b>${row('point',i+1+' / '+N)}${row(D.metric,f3(a.hist[i],5))}`,e); };
    hit.onmouseleave=()=>{cur.setAttribute('opacity',0);untip();};
    g.appendChild(cur); g.appendChild(hit);
    g.appendChild(txt(S('text',{class:'tick',x:(ML+W-MR)/2,y:H-8,'text-anchor':'middle'}),
      'logged step'));
    f.appendChild(g); host.appendChild(f); });
}

function ridge(){
  const host=document.getElementById('ridge'); host.innerHTML='';
  const A=on().filter(a=>a.gate.length>4);
  if(!A.length){ document.getElementById('ridgecard').classList.add('hidden'); return; }
  document.getElementById('ridgecard').classList.remove('hidden');
  const W=1180,ROW=64,ML=170,MR=30,MT=28,MB=36, H=MT+A.length*ROW+MB;
  const vals=A.flatMap(a=>a.gate);
  const x0=Math.min(0,...vals), x1=Math.max(1.05,...vals);
  const X=v=>ML+(v-x0)/((x1-x0)||1)*(W-ML-MR);
  const g=S('svg',{viewBox:`0 0 ${W} ${H}`});
  [[0.5,'sigmoid ≈ 0.5'],[1.0,'r·softmax = 1.0']].forEach(([v,l])=>{
    if(v<x0||v>x1) return;
    g.appendChild(S('line',{x1:X(v),x2:X(v),y1:MT-8,y2:H-MB,stroke:'var(--ink-3)','stroke-width':1}));
    g.appendChild(txt(S('text',{class:'tick',x:X(v)+5,y:MT-12}),l)); });
  A.forEach((a,i)=>{ const yb=MT+(i+1)*ROW, c=col(a), NB=90, h=new Array(NB).fill(0);
    a.gate.forEach(v=>{ const k=Math.floor((v-x0)/((x1-x0)||1)*NB); if(k>=0&&k<NB) h[k]++; });
    const mx=Math.max(...h)||1; let p='';
    h.forEach((n,k)=>{ p+=(k?'L':'M')+X(x0+(k+.5)/NB*(x1-x0))+','+(yb-n/mx*(ROW-12)); });
    g.appendChild(S('path',{d:`M${X(x0)},${yb} `+p+` L${X(x1)},${yb}Z`,fill:c,'fill-opacity':.30,
      stroke:c,'stroke-width':1.5,'stroke-linejoin':'round'}));
    g.appendChild(S('line',{class:'gr',x1:ML,x2:W-MR,y1:yb,y2:yb}));
    const m=a.gate.reduce((p2,q)=>p2+q,0)/a.gate.length;
    g.appendChild(S('line',{x1:X(m),x2:X(m),y1:yb,y2:yb-20,stroke:c,'stroke-width':2.4}));
    g.appendChild(txt(S('text',{class:'lab',x:ML-12,y:yb-4,'text-anchor':'end',fill:c}),a.name));
    const hit=S('rect',{x:ML,y:yb-ROW+4,width:W-ML-MR,height:ROW,fill:'transparent'});
    hit.style.cursor='crosshair';
    hit.onmousemove=e=>{ const r=g.getBoundingClientRect();
      const gv=x0+((e.clientX-r.left)/r.width*W-ML)/(W-ML-MR)*(x1-x0);
      const k=Math.max(0,Math.min(NB-1,Math.floor((gv-x0)/((x1-x0)||1)*NB)));
      tip(`<b>${a.name}</b>${row('gate value',gv.toFixed(2))}${row('samples in bin',h[k])}`+
        `${row('mean',m.toFixed(3))}`,e); };
    hit.onmouseleave=untip; g.appendChild(hit); });
  for(let i=0;i<=4;i++){ const v=x0+(x1-x0)*i/4;
    g.appendChild(txt(S('text',{class:'tick',x:X(v),y:H-14,'text-anchor':'middle'}),v.toFixed(2))); }
  g.appendChild(txt(S('text',{x:(ML+W-MR)/2,y:H-1,'text-anchor':'middle'}),'gate value g'));
  host.appendChild(g);
}

function kpis(){
  const A=on(), withCtrl=A.filter(a=>a.ctrl);
  const res=withCtrl.filter(a=>{ const c=D.arms.find(x=>x.name===a.ctrl);
    return c&&c.mean!=null&&a.mean!=null&&D.mde!=null&&Math.abs(a.mean-c.mean)>=D.mde; });
  const scored=A.filter(a=>a.mean!=null);
  const best=scored.length?scored.reduce((p,q)=>q.mean>p.mean?q:p):null;
  const tiles=[['Arms shown',`${A.length} / ${D.arms.length}`,'filter row scopes every chart'],
    ['Contrasts resolved',`${res.length} / ${withCtrl.length}`,
      D.mde==null?'no floor estimable':'|effect| ≥ MDE '+f3(D.mde)],
    ['Best success rate',best?f3(best.mean):'--',best?best.name:'no scored arm'],
    ['Detection floor',D.mde==null?'--':'±'+f3(D.mde),'paired t, n=3, 80% power']];
  document.getElementById('kpis').innerHTML = tiles.map(([k,v,d])=>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join('');
}

function table(){
  const A=on();
  document.getElementById('tblbody').innerHTML='<table><thead><tr><th>arm</th><th>family</th>'+
    '<th>seeds</th><th>mean</th><th>control</th><th>effect</th><th>resolved</th></tr></thead><tbody>'+
    A.map(a=>{ const c=D.arms.find(x=>x.name===a.ctrl);
      const e=(c&&c.mean!=null&&a.mean!=null)?a.mean-c.mean:null;
      return `<tr><td style="color:${col(a)}">${a.name}</td><td>${a.family}</td>`+
        `<td>${a.seeds.map(s=>f3(s)).join(', ')||'--'}</td><td>${f3(a.mean)}</td>`+
        `<td>${a.ctrl||'--'}</td><td>${e==null?'--':(e>0?'+':'')+f3(e)}</td>`+
        `<td>${e==null||D.mde==null?'--':(Math.abs(e)>=D.mde?'yes':'underpowered')}</td></tr>`;
    }).join('')+'</tbody></table>';
}
function draw(){ kpis(); estimation(); small(); ridge(); table(); }
draw();
</script></body></html>
"""


def render(payload, subtitle="", refresh=0):
    """The whole page as one string. No file IO, so it is trivially testable.

    `refresh` seconds > 0 emits a meta-refresh, so a browser left open on the report
    picks up each regeneration on its own. A meta tag rather than a fetch loop: the
    file is opened from a filesystem path as often as over HTTP, and fetch() against
    file:// is blocked by CORS in every browser that matters.
    """
    import datetime as _dt

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head = (HEAD.replace("__CSS__", _css(payload["tokens"]))
                .replace("__SUB__", subtitle or f"{len(payload['arms'])} arms · "
                                                f"{len(payload['contrasts'])} contrasts")
                .replace("__METRIC__", payload["metric"])
                .replace("__STAMP__", stamp)
                .replace("__LIVE__", f" &middot; refreshing every {int(refresh)}s"
                                     if refresh else "")
                .replace("__REFRESH__",
                         f'<meta http-equiv="refresh" content="{int(refresh)}">\n'
                         if refresh else ""))
    return head.replace("__DATA__", json.dumps(payload, allow_nan=False)) + JS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", help="campaign.json, same file analysis/figures.py reads")
    ap.add_argument("--runs", help="glob of run dirs holding wandb_history.csv")
    ap.add_argument("--metric", default="train_reg_loss", help="metric for the facet grid")
    ap.add_argument("--out", default="figures/report.html")
    ap.add_argument("--selftest", action="store_true",
                    help="render from the same synthetic campaign figures.py uses")
    ap.add_argument("--refresh", type=int, default=0, metavar="SECONDS",
                    help="emit a meta-refresh so an open tab updates itself")
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="regenerate every N seconds until interrupted (implies "
                         "--refresh N); this is the live-report mode")
    a = ap.parse_args(argv)
    if a.watch and not a.refresh:
        a.refresh = a.watch

    if a.selftest:
        # the same synthetic campaign the PNG suite exercises, so the two media are
        # verified against identical inputs and any divergence is a real difference
        import tempfile
        tmp = FG._synth(Path(tempfile.mkdtemp(prefix="lpwm_report_")))
        camp = json.loads((tmp / "campaign.json").read_text())
        arms, gates, gv = camp.get("arms", {}), camp.get("gates", []), camp.get("gate_values", {})
        runs = FG.load_runs(str(tmp / "runs" / "*"))
    else:
        camp = json.loads(Path(a.campaign).read_text()) if a.campaign else {}
        arms, gates, gv = camp.get("arms", {}), camp.get("gates", []), camp.get("gate_values", {})
        runs = FG.load_runs(a.runs) if a.runs else []
    if not arms and not runs:
        ap.error("nothing to report: pass --campaign and/or --runs (or --selftest)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def once():
        # re-read the campaign each pass: a live report whose inputs are snapshotted
        # at start-up is just a slower static report
        c = json.loads(Path(a.campaign).read_text()) if a.campaign and not a.selftest else camp
        r = FG.load_runs(a.runs) if (a.runs and not a.selftest) else runs
        payload = build_payload(c.get("arms", {}), c.get("gates", []),
                                c.get("gate_values", {}), r, metric=a.metric)
        # write via a temp file + rename: a browser that reloads mid-write would
        # otherwise render a truncated page
        tmp = out.with_suffix(".html.tmp")
        tmp.write_text(render(payload, refresh=a.refresh))
        tmp.replace(out)
        return payload

    payload = once()
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(payload['arms'])} arms, {len(payload['contrasts'])} contrasts"
          + (f", refreshing every {a.refresh}s" if a.refresh else "") + ")")
    if a.watch:
        import time

        print(f"watching: regenerating every {a.watch}s (Ctrl-C to stop)")
        try:
            while True:
                time.sleep(a.watch)
                payload = once()
                print(f"  refreshed {out} at {time.strftime('%H:%M:%S')} "
                      f"({len(payload['arms'])} arms)")
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
