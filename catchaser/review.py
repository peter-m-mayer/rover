"""Cat Chaser 3000 — label review & triage tool.

A phone-and-computer web tool for turning the harness's weak pre-labels into a
clean dataset with the least possible effort. It does NOT draw boxes (precise
box editing belongs in labelImg / Roboflow) — it triages, which is the 80% that
can be single taps on a couch:

  Good   — the detector's box is fine → keep the label as-is
  No cat — no cat in this frame (or the box is junk) → label forced EMPTY, so
           the frame becomes a clean NEGATIVE instead of a bad positive
  Fix    — a boxed cat whose box is wrong → copied into review_fix/ to re-box
  Missed — the detector drew NO box but a cat IS there (false negative) →
           copied into review_fix/ to be boxed; NOT left as a poison negative

The button set is context-aware: frames the detector boxed offer Good/No-cat/
Fix; frames with no box offer No-cat/Missed. This closes the old trap where
"Good" on a no-box frame silently stored a present cat as a negative.

Every decision is saved to review_status.json (with a snapshot of the original
label), so you can stop/resume anytime and Undo is lossless.

Runs anywhere with Flask + OpenCV — your computer (point it at a copied dataset)
or the Pi. No hardware needed.

    python3 -m catchaser.review datasets/livingroom_evening
    # open http://localhost:5000  (or http://<pi-ip>:5000 from your phone)

Keyboard (desktop): g=Good  n=No cat  f=Fix  u=Undo  ←/→ prev/next
"""

import argparse
import glob
import json
import os
import shutil
import sys


def _load_status(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


class ReviewStore:
    """Dataset triage state; pure file operations, unit-testable without Flask."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.img_dir = os.path.join(dataset, "images")
        self.lbl_dir = os.path.join(dataset, "labels")
        self.fix_img = os.path.join(dataset, "review_fix", "images")
        self.fix_lbl = os.path.join(dataset, "review_fix", "labels")
        self.status_path = os.path.join(dataset, "review_status.json")
        self.status = _load_status(self.status_path)   # frame -> {"decision", "orig"}
        self.frames = [os.path.basename(p)
                       for p in sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))]

    # ---- helpers ------------------------------------------------------------
    def _label_path(self, frame):
        return os.path.join(self.lbl_dir, os.path.splitext(frame)[0] + ".txt")

    def _read_label(self, frame):
        p = self._label_path(frame)
        return open(p).read() if os.path.exists(p) else ""

    def _write_label(self, frame, text):
        p = self._label_path(frame)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)

    def _snapshot_orig(self, frame):
        """Record the original label once, so Undo is lossless."""
        if frame not in self.status or "orig" not in self.status[frame]:
            self.status.setdefault(frame, {})["orig"] = self._read_label(frame)

    def _save(self):
        with open(self.status_path, "w") as f:
            json.dump(self.status, f, indent=0)

    # ---- API ----------------------------------------------------------------
    _DECISIONS = ("good", "nocat", "fix", "missed")

    def counts(self):
        c = {"total": len(self.frames), "good": 0, "nocat": 0, "fix": 0,
             "missed": 0, "pending": 0}
        for fr in self.frames:
            d = self.status.get(fr, {}).get("decision")
            c[d if d in self._DECISIONS else "pending"] += 1
        return c

    def decision(self, frame):
        return self.status.get(frame, {}).get("decision")

    def is_boxed(self, frame):
        """Did the detector draw a box here? (drives the context-aware UI.)"""
        st = self.status.get(frame, {})
        src = st["orig"] if "orig" in st else self._read_label(frame)
        return bool(src.strip())

    def decide(self, frame, decision):
        if frame not in self.frames:
            raise KeyError(frame)
        if decision not in self._DECISIONS + ("undo",):
            raise ValueError(decision)
        self._snapshot_orig(frame)
        orig = self.status[frame]["orig"]

        if decision == "undo":
            self._write_label(frame, orig)                 # restore original label
            self._cleanup_fix(frame)
            self.status[frame].pop("decision", None)
        elif decision == "good":
            self._write_label(frame, orig)                 # keep detector's box
            self._cleanup_fix(frame)
            self.status[frame]["decision"] = "good"
        elif decision == "nocat":
            self._write_label(frame, "")                   # force negative
            self._cleanup_fix(frame)
            self.status[frame]["decision"] = "nocat"
        elif decision in ("fix", "missed"):
            # Both route to review_fix/ for real boxing. "fix" carries the
            # detector's (wrong) box to adjust; "missed" starts from empty.
            os.makedirs(self.fix_img, exist_ok=True)
            os.makedirs(self.fix_lbl, exist_ok=True)
            shutil.copy(os.path.join(self.img_dir, frame),
                        os.path.join(self.fix_img, frame))
            self._write_label(frame, orig)
            with open(os.path.join(self.fix_lbl,
                                   os.path.splitext(frame)[0] + ".txt"), "w") as f:
                f.write(orig)
            self.status[frame]["decision"] = decision
        self._save()
        return self.decision(frame)

    def resurface_traps(self):
        """Reopen frames mislabeled by the old trap: 'good' on a no-box frame.

        Those affirmed a cat but kept an empty label (a poison negative).
        Clears their decision so they return to the queue for proper
        re-triage (where 'Missed' now routes them to boxing). Returns count.
        """
        n = 0
        for fr in self.frames:
            st = self.status.get(fr, {})
            if st.get("decision") == "good" and not (st.get("orig") or "").strip():
                st.pop("decision", None)
                n += 1
        if n:
            self._save()
        return n

    def _cleanup_fix(self, frame):
        for d, ext in ((self.fix_img, ".jpg"), (self.fix_lbl, ".txt")):
            p = os.path.join(d, os.path.splitext(frame)[0] + ext) if ext == ".txt" \
                else os.path.join(d, frame)
            if os.path.exists(p):
                os.remove(p)

    def annotated_jpeg(self, frame):
        import cv2
        import numpy as np
        img = cv2.imread(os.path.join(self.img_dir, frame))
        if img is None:
            return None
        h, w = img.shape[:2]
        for line in self._read_label(frame).strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else None


_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Cat label review</title><style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee;text-align:center}
 img{width:100%;max-width:720px;background:#000}
 .bar{padding:6px;font-size:.95em;background:#1c1c1c}
 .tag{display:inline-block;padding:2px 8px;border-radius:8px;margin-left:6px}
 button{font-size:1.25em;padding:16px;margin:5px;width:44%;border:0;border-radius:12px;color:#fff}
 .good{background:#2e7d32}.nocat{background:#455a64}.fix{background:#e65100}
 .missed{background:#c62828}
 .undo{background:#6a1b9a;width:44%}.nav{background:#333;width:44%}
 .grid{display:flex;flex-wrap:wrap;justify-content:center}
 .hint{font-size:.8em;color:#888;padding:2px}
</style></head><body>
<div class=bar>
 <b id=pos>–</b>/<b id=tot>–</b> &nbsp; <span id=cur class=tag>?</span>
 &nbsp;|&nbsp; good <b id=cg>0</b> · nocat <b id=cn>0</b> · fix <b id=cf>0</b>
 · missed <b id=cm>0</b> · left <b id=cp>0</b>
</div>
<img id=view src="">
<div class=hint id=hint></div>
<div class=grid id=buttons></div>
<script>
let items=[],i=0;
const BOXED=[  // frame the detector boxed
 ['good','✓ Good (g)'],['nocat','✗ No cat (n)'],['fix','✎ Fix box (f)']];
const NOBOX=[  // detector drew nothing
 ['nocat','✗ No cat (n)'],['missed','🐾 Missed! (m)']];
async function load(){let r=await(await fetch('/api/list')).json();
 items=r.items;refreshCounts(r.counts);
 let p=items.findIndex(x=>!x.decision);i=p<0?0:p;show();}
function show(){if(!items.length)return;let it=items[i];
 document.getElementById('view').src='/img/'+encodeURIComponent(it.frame)+'?'+Date.now();
 document.getElementById('pos').textContent=i+1;
 document.getElementById('tot').textContent=items.length;
 let c=document.getElementById('cur');c.textContent=it.decision||'pending';
 c.style.background={good:'#2e7d32',nocat:'#455a64',fix:'#e65100',
   missed:'#c62828'}[it.decision]||'#b71c1c';
 document.getElementById('hint').textContent=it.boxed
   ? 'detector boxed a cat — is the box right?'
   : 'detector saw no cat — is that correct?';
 let set=it.boxed?BOXED:NOBOX,html='';
 for(const [d,label] of set) html+=`<button class="${d}" onclick="decide('${d}')">${label}</button>`;
 html+=`<button class=undo onclick="decide('undo')">↶ Undo (u)</button>`;
 html+=`<button class=nav onclick="move(-1)">◀ Prev</button>`;
 html+=`<button class=nav onclick="move(1)">Next ▶</button>`;
 document.getElementById('buttons').innerHTML=html;}
function refreshCounts(c){cg.textContent=c.good;cn.textContent=c.nocat;
 cf.textContent=c.fix;cm.textContent=c.missed;cp.textContent=c.pending;tot.textContent=c.total;}
async function decide(d){let it=items[i];
 // guard: keyboard shortcut must match the current frame's button set
 let valid=(it.boxed?['good','nocat','fix']:['nocat','missed']).concat('undo');
 if(!valid.includes(d))return;
 let r=await(await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({frame:it.frame,decision:d})}).then(x=>x.json()));
 it.decision=r.decision;refreshCounts(r.counts);
 if(d!=='undo'){let n=items.findIndex((x,k)=>k>i&&!x.decision);i=n<0?Math.min(i+1,items.length-1):n;}
 show();}
function move(d){i=Math.max(0,Math.min(items.length-1,i+d));show();}
document.onkeydown=e=>{let k=e.key.toLowerCase();
 if(k==='g')decide('good');else if(k==='n')decide('nocat');
 else if(k==='f')decide('fix');else if(k==='m')decide('missed');
 else if(k==='u')decide('undo');
 else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);};
load();
</script></body></html>"""


def make_app(store):
    from flask import Flask, Response, request, jsonify
    app = Flask(__name__)

    @app.route("/")
    def index():
        return _PAGE

    @app.route("/api/list")
    def api_list():
        return jsonify({
            "items": [{"frame": fr, "decision": store.decision(fr),
                       "boxed": store.is_boxed(fr)} for fr in store.frames],
            "counts": store.counts(),
        })

    @app.route("/img/<path:frame>")
    def img(frame):
        data = store.annotated_jpeg(frame)
        if data is None:
            return Response(status=404)
        return Response(data, mimetype="image/jpeg")

    @app.route("/api/decide", methods=["POST"])
    def decide():
        body = request.get_json(force=True)
        try:
            dec = store.decide(body["frame"], body["decision"])
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"decision": dec, "counts": store.counts()})

    return app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cat label review & triage")
    p.add_argument("dataset", help="datasets/<session> directory")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--fix-traps", action="store_true",
                   help="reopen frames mislabeled by the old 'good on no-box' "
                        "trap so you can re-triage them as Missed, then exit")
    args = p.parse_args(argv)

    if not os.path.isdir(os.path.join(args.dataset, "images")):
        print(f"[review] no images/ under {args.dataset}", file=sys.stderr)
        return 1
    store = ReviewStore(args.dataset)
    if args.fix_traps:
        n = store.resurface_traps()
        print(f"[review] reopened {n} trap frame(s) ('good' on no-box) for "
              f"re-triage — run again without --fix-traps and press Missed.")
        return 0
    print(f"[review] {len(store.frames)} frames in {args.dataset}")
    print(f"[review] open http://<this-host>:{args.port}   (g/n/f/u + arrows)")
    make_app(store).run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
