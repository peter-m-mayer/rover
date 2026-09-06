#!/usr/bin/env python3
"""Build the Cat Chaser 3000 presentation (single-file HTML, images inlined).

Usage: python3 docs/build_presentation.py [--images /tmp/dets] [--out docs/catchaser_presentation.html]
"""

import argparse
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(images_dir, name):
    p = os.path.join(images_dir, name)
    with open(p, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def build(images_dir, out_path, video_path=None):
    img = {k: b64(images_dir, f"det_{k}.jpg")
           for k in ("001", "008", "010", "011", "012", "013")}

    html = HTML_TEMPLATE
    for k, v in img.items():
        html = html.replace("{{IMG_" + k + "}}", v)

    if video_path and os.path.exists(video_path):
        with open(video_path, "rb") as f:
            vid = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
        html = html.replace("{{VIDEO}}", vid)
    else:
        # No video available: drop the whole video slide cleanly.
        import re
        html = re.sub(r"<!-- VIDEO_SLIDE_START -->.*?<!-- VIDEO_SLIDE_END -->",
                      "", html, flags=re.S)

    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({os.path.getsize(out_path)//1024} KB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAT CHASER 3000 — Field Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;600;700&family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#070b07; --panel:#0c120b; --panel2:#101810;
  --green:#6cff5c; --green-dim:#3d9a45; --green-dark:#1c3d20;
  --amber:#ffb454; --red:#ff5470; --cyan:#5ee6d0; --blue:#6ea8ff;
  --text:#c9dfc2; --text-dim:#7d967a; --line:#22331f;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden;background:var(--bg)}
body{font-family:'IBM Plex Mono',monospace;color:var(--text);font-size:16px}

/* ambient: scanlines + vignette + grain */
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:50;
  background:repeating-linear-gradient(0deg,rgba(0,0,0,.22) 0 1px,transparent 1px 3px);}
body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:51;
  background:radial-gradient(ellipse at 50% 45%,transparent 55%,rgba(0,0,0,.55) 100%);}
.grain{position:fixed;inset:-50%;pointer-events:none;z-index:49;opacity:.05;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)'/%3E%3C/svg%3E");
  animation:grain 8s steps(10) infinite}
@keyframes grain{0%,100%{transform:translate(0,0)}20%{transform:translate(-3%,2%)}40%{transform:translate(2%,-3%)}60%{transform:translate(-2%,-2%)}80%{transform:translate(3%,3%)}}

.deck{position:relative;height:100%;width:100%}
.slide{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;
  padding:5.5vh 7vw 11vh;opacity:0;transform:translateX(60px);transition:opacity .45s ease,transform .45s ease;
  pointer-events:none;overflow:hidden}
.slide.active{opacity:1;transform:translateX(0);pointer-events:auto}
.slide.prev{transform:translateX(-60px)}

.kicker{font-size:.72rem;letter-spacing:.35em;color:var(--green-dim);text-transform:uppercase;margin-bottom:1.1rem}
.kicker b{color:var(--green);font-weight:500}
h1,h2{font-family:'Chakra Petch',sans-serif;line-height:1.02}
h2{font-size:clamp(1.6rem,4.2vw,2.9rem);font-weight:700;color:#eaffe6;margin-bottom:1.6rem}
h2 .accent{color:var(--green)}
p.lede{color:var(--text-dim);max-width:62ch;line-height:1.65;font-size:.95rem}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:2.5rem;align-items:start}
@media(max-width:860px){.columns{grid-template-columns:1fr;gap:1.2rem}.slide{padding:4vh 6vw 12vh}}

/* title */
.title-slide{align-items:flex-start;justify-content:center}
.title-big{font-family:'Chakra Petch',sans-serif;font-weight:700;
  font-size:clamp(3rem,10vw,7.5rem);color:#eaffe6;letter-spacing:.01em;line-height:.95;
  text-shadow:0 0 22px rgba(108,255,92,.25)}
.title-big .num{color:transparent;-webkit-text-stroke:2px var(--green);}
.title-sub{margin-top:1.4rem;color:var(--text-dim);font-size:.95rem;max-width:56ch;line-height:1.7}
.title-meta{margin-top:2.2rem;display:flex;gap:1rem;flex-wrap:wrap}
.chip{border:1px solid var(--line);background:var(--panel);padding:.45rem .8rem;font-size:.72rem;
  letter-spacing:.12em;color:var(--text-dim)}
.chip b{color:var(--green);font-weight:500}
.radar{position:absolute;right:6vw;top:50%;transform:translateY(-50%);width:min(34vw,340px);aspect-ratio:1;
  border-radius:50%;border:1px solid var(--green-dark);
  background:
    radial-gradient(circle,transparent 0 24%,rgba(108,255,92,.05) 25% 26%,transparent 27% 49%,rgba(108,255,92,.05) 50% 51%,transparent 52% 74%,rgba(108,255,92,.05) 75% 76%,transparent 77%),
    linear-gradient(0deg,transparent 49.6%,rgba(108,255,92,.12) 50%,transparent 50.4%),
    linear-gradient(90deg,transparent 49.6%,rgba(108,255,92,.12) 50%,transparent 50.4%);
  overflow:hidden;opacity:.9}
.radar::before{content:"";position:absolute;inset:0;border-radius:50%;
  background:conic-gradient(from 0deg,rgba(108,255,92,.35),transparent 70deg,transparent);
  animation:sweep 4s linear infinite}
@keyframes sweep{to{transform:rotate(360deg)}}
.blip{position:absolute;width:10px;height:10px;border-radius:50%;background:var(--amber);
  left:63%;top:31%;box-shadow:0 0 12px var(--amber);animation:blip 4s linear infinite}
@keyframes blip{0%,7%{opacity:1}30%,88%{opacity:.15}96%,100%{opacity:1}}
.blip::after{content:"CAT";position:absolute;left:14px;top:-4px;font-size:.6rem;letter-spacing:.2em;color:var(--amber)}
@media(max-width:860px){.radar{display:none}}

/* content bits */
ul.sys{list-style:none;display:flex;flex-direction:column;gap:.65rem;font-size:.88rem}
ul.sys li{padding-left:1.4rem;position:relative;line-height:1.5}
ul.sys li::before{content:"▸";position:absolute;left:0;color:var(--green)}
ul.sys li b{color:#eaffe6;font-weight:600}
ul.sys li .d{color:var(--text-dim)}

.term{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--green-dark);
  padding:1rem 1.2rem;font-size:.76rem;line-height:1.75;overflow-x:auto;white-space:pre;color:var(--text-dim)}
.term .g{color:var(--green)} .term .a{color:var(--amber)} .term .r{color:var(--red)}
.term .c{color:var(--cyan)} .term .w{color:#eaffe6} .term .dim{color:#54644f}
.term-title{font-size:.66rem;letter-spacing:.25em;color:var(--green-dim);border:1px solid var(--line);
  border-bottom:none;background:var(--panel2);padding:.45rem 1.2rem;display:flex;gap:.5rem;align-items:center}
.term-title::before{content:"●";color:var(--red);font-size:.6rem}

table.spec{border-collapse:collapse;width:100%;font-size:.82rem}
table.spec td,table.spec th{border-bottom:1px solid var(--line);padding:.5rem .6rem;text-align:left;vertical-align:top}
table.spec th{color:var(--green-dim);font-weight:500;font-size:.7rem;letter-spacing:.2em;text-transform:uppercase}
table.spec td:first-child{color:#eaffe6}
table.spec td .d{color:var(--text-dim)}
.big-num{font-family:'Chakra Petch',sans-serif;font-weight:700;font-size:2.6rem;color:var(--green);line-height:1}
.big-num small{font-size:1rem;color:var(--text-dim);font-family:'IBM Plex Mono',monospace}
.stat-row{display:flex;gap:2.6rem;flex-wrap:wrap;margin:1.4rem 0}
.stat-label{font-size:.68rem;letter-spacing:.2em;color:var(--text-dim);text-transform:uppercase;margin-top:.4rem}

/* HUD photos */
.photo-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1.1rem}
.hud{position:relative;background:var(--panel)}
.hud img{width:100%;display:block;filter:saturate(.92) contrast(1.03)}
.hud .corners{position:absolute;inset:0;pointer-events:none}
.hud .corners i{position:absolute;width:14px;height:14px;border:2px solid var(--green)}
.hud .corners i:nth-child(1){top:-2px;left:-2px;border-right:none;border-bottom:none}
.hud .corners i:nth-child(2){top:-2px;right:-2px;border-left:none;border-bottom:none}
.hud .corners i:nth-child(3){bottom:-2px;left:-2px;border-right:none;border-top:none}
.hud .corners i:nth-child(4){bottom:-2px;right:-2px;border-left:none;border-top:none}
.hud figcaption{display:flex;justify-content:space-between;gap:.6rem;padding:.5rem .65rem;font-size:.68rem;
  color:var(--text-dim);border:1px solid var(--line);border-top:none;letter-spacing:.05em}
.hud .conf{color:var(--green)} .hud .conf.bad{color:var(--red)}

.states{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}
.state-card{border:1px solid var(--line);background:var(--panel);padding:1rem 1.1rem}
.state-card h3{font-family:'Chakra Petch',sans-serif;font-size:1rem;letter-spacing:.12em;margin-bottom:.5rem;
  display:flex;align-items:center;gap:.55rem}
.led{width:10px;height:10px;border-radius:50%;display:inline-block;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.45}}
.state-card p{font-size:.78rem;color:var(--text-dim);line-height:1.6}

/* confusion matrix */
.cm{display:grid;grid-template-columns:auto 130px 130px;grid-template-rows:auto 108px 108px;gap:6px}
.cm-corner{}
.cm-head{display:flex;align-items:center;justify-content:center;text-align:center;font-size:.66rem;
  letter-spacing:.16em;color:var(--green-dim);text-transform:uppercase;line-height:1.35;padding:.3rem}
.cm-cell{display:flex;flex-direction:column;align-items:center;justify-content:center;
  border:1px solid var(--line);background:var(--panel)}
.cm-cell b{font-family:'Chakra Petch',sans-serif;font-size:2.3rem;line-height:1}
.cm-cell span{font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;color:var(--text-dim);margin-top:.35rem}
.cm-cell.tp{border-color:var(--green)} .cm-cell.tp b{color:var(--green)}
.cm-cell.tn{border-color:var(--green-dark)} .cm-cell.tn b{color:var(--green-dim)}
.cm-cell.fp{border-color:var(--line)} .cm-cell.fp b{color:var(--text-dim)}
.cm-cell.fn{border-color:var(--amber)} .cm-cell.fn b{color:var(--amber)}
.metric-row{display:flex;gap:2.2rem;flex-wrap:wrap;margin-bottom:1.2rem}
.metric b{font-family:'Chakra Petch',sans-serif;font-size:2.4rem;color:var(--green);line-height:1;display:block}
.metric.warn b{color:var(--amber)}
.metric span{font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:var(--text-dim)}
.metric small{font-size:.9rem;color:var(--text-dim);font-family:'IBM Plex Mono',monospace}

/* chrome */
.hudbar{position:fixed;left:0;right:0;bottom:0;display:flex;justify-content:space-between;align-items:center;
  padding:.8rem 2rem;font-size:.7rem;letter-spacing:.15em;color:var(--text-dim);z-index:60;
  background:linear-gradient(0deg,rgba(7,11,7,.95),transparent)}
.hudbar .fcount{color:var(--green)}
.hudbar .statechip{padding:.25rem .7rem;border:1px solid var(--line);color:var(--bg);font-weight:600;letter-spacing:.2em}
.dots{display:flex;gap:.45rem}
.dots i{width:6px;height:6px;background:var(--green-dark);cursor:pointer;transition:.2s}
.dots i.on{background:var(--green);box-shadow:0 0 8px rgba(108,255,92,.6)}
.navhint{position:fixed;top:1rem;right:1.4rem;font-size:.64rem;letter-spacing:.2em;color:#54644f;z-index:60}
.arrow{position:fixed;top:50%;transform:translateY(-50%);z-index:60;background:none;border:1px solid var(--line);
  color:var(--green-dim);font-family:inherit;font-size:1rem;padding:.7rem .8rem;cursor:pointer;transition:.2s}
.arrow:hover{color:var(--green);border-color:var(--green-dim)}
.arrow.left{left:.9rem}.arrow.right{right:.9rem}
@media(max-width:860px){.arrow{display:none}.hudbar{padding:.7rem 1rem}}
.tick{position:fixed;top:0;left:0;right:0;height:2px;background:var(--green-dark);z-index:61}
.tick i{display:block;height:100%;background:var(--green);width:0;transition:width .4s ease}
</style>
</head>
<body>
<div class="grain"></div>
<div class="tick"><i id="tick"></i></div>
<div class="navhint">←/→ · SPACE · SWIPE</div>

<div class="deck" id="deck">

<!-- 1 · TITLE -->
<section class="slide title-slide" data-state="BOOT" data-color="#6cff5c">
  <div class="radar"><span class="blip"></span></div>
  <div class="kicker">FIELD REPORT · 2026-09-05 → 07 · <b>UNIT: YAHBOOM RASPBOT-V2</b></div>
  <div class="title-big">CAT CHASER<br><span class="num">3000</span></div>
  <p class="title-sub">An autonomous cat-pursuit robot: pretrained neural detector, PID mecanum
  control, ultrasonic safety, and one extremely unimpressed Bengal.
  Built across cloud + local Claude Code sessions, validated in PyBullet, then let loose on the carpet.</p>
  <div class="title-meta">
    <span class="chip">PI 5 · <b>AARCH64</b></span>
    <span class="chip">DETECTOR · <b>YOLOV8N/COCO</b></span>
    <span class="chip">LOOP · <b>~9 HZ</b></span>
    <span class="chip">TESTS · <b>239 GREEN</b></span>
    <span class="chip">CATS HARMED · <b>0</b></span>
  </div>
</section>

<!-- 2 · MISSION PIPELINE -->
<section class="slide" data-state="PLAN" data-color="#5ee6d0">
  <div class="kicker">// 01 · MISSION ARCHITECTURE</div>
  <h2>One loop, five links, <span class="accent">zero mercy</span></h2>
  <div class="term-title">PERCEPTION → DECISION → ACTUATION</div>
  <div class="term">
<span class="w">camera</span> 640×480 BGR
   ↓
<span class="g">YOLOv8n ONNX</span> @320px ──► cat boxes (COCO class 15), conf ≥ 0.50
   ↓
<span class="w">centroid → heading error</span>   err = (cx − W/2)/(W/2) ∈ [−1, +1]
   ↓
<span class="g">PID</span> (KP=40 · KI=0 · KD=6, hardware-calibrated) ──► turn
   ↓
<span class="w">mecanum mix</span> drive(forward, turn) → 4 wheel speeds, per-wheel cap 120
   ↓
<span class="a">safety layer</span>: ultrasonic HOLD @200mm · IR-remote kill · Ctrl+C · runtime limit</div>
  <p class="lede" style="margin-top:1.2rem">Same controller runs against real hardware and the PyBullet
  digital twin — the sim caught the design bugs, the carpet caught the physics bugs.</p>
</section>

<!-- 3 · PLATFORM -->
<section class="slide" data-state="HW" data-color="#6ea8ff">
  <div class="kicker">// 02 · THE PLATFORM</div>
  <h2>Yahboom RASPBOT-V2 <span class="accent">+ Raspberry Pi 5</span></h2>
  <div class="columns">
    <table class="spec">
      <tr><th>Part</th><th>Detail</th></tr>
      <tr><td>Compute</td><td class="d">Pi 5 Model B rev 1.1 · aarch64 · Python 3.11</td></tr>
      <tr><td>MCU bridge</td><td class="d">I2C bus 1 @ 0x2B — motors, servos, LEDs, buzzer</td></tr>
      <tr><td>Drive</td><td class="d">4× mecanum wheels, ±255, <i>no encoders</i></td></tr>
      <tr><td>Camera</td><td class="d">USB 640×480, fixed focus (~0.4 m → ∞)</td></tr>
      <tr><td>Ranging</td><td class="d">Forward ultrasonic, mm resolution</td></tr>
      <tr><td>Status</td><td class="d">14× WS2812B LEDs + buzzer + IR remote (kill switch)</td></tr>
    </table>
    <div>
      <p class="lede">“Motor commands are suggestions” — no encoders means the camera
      is the only truth. The vendor driver's real API had to be discovered by
      introspection on-device: the docs promised <b style="color:#eaffe6">Ctrl_WS2812B</b>;
      the hardware shipped <b style="color:#eaffe6">Ctrl_WQ2812_ALL(state, palette_index)</b>.</p>
      <div class="stat-row">
        <div><div class="big-num">0x2B</div><div class="stat-label">MCU · I2C addr</div></div>
        <div><div class="big-num">0xFF</div><div class="stat-label">IR idle byte (not 0!)</div></div>
        <div><div class="big-num">4<small>/4</small></div><div class="stat-label">wheels omnidirectional</div></div>
      </div>
    </div>
  </div>
</section>

<!-- 4 · NEURAL NET -->
<section class="slide" data-state="NN" data-color="#6cff5c">
  <div class="kicker">// 03 · THE NEURAL NET · PROVENANCE</div>
  <h2>YOLOv8-nano, <span class="accent">COCO-raised</span>, cat-filtered</h2>
  <div class="columns">
    <div>
      <ul class="sys">
        <li><b>Origin:</b> <span class="d">Ultralytics YOLOv8n, pretrained on COCO 2017 — 118k images, 80 classes. We keep exactly one: class 15, <i>cat</i>.</span></li>
        <li><b>Export:</b> <span class="d">yolov8n.pt → ONNX · 320 px input · opset 12 · simplified · 12.1 MB · ~3.2M params</span></li>
        <li><b>Runtime:</b> <span class="d">pure onnxruntime + OpenCV on the robot — no ultralytics, no torch. Letterbox → NCHW → NMS in ~150 lines.</span></li>
        <li><b>Threshold:</b> <span class="d">conf ≥ 0.50, calibrated from field data (real cat ≥ 0.53 always; false positives 0.41–0.45).</span></li>
        <li><b>Fine-tune?</b> <span class="d">Not yet — 100% precision / 90% recall over 244 triaged field frames (see the confusion matrix). The misses are the fine-tune fuel when we get there.</span></li>
      </ul>
    </div>
    <div>
      <div class="term-title">VALIDATION LADDER</div>
      <div class="term">
<span class="dim">x86 sanity</span>   real-cat photo      <span class="g">conf 0.94</span> ✓
<span class="dim">x86 bench</span>    21 ms/frame         <span class="g">47 Hz</span> ✓
<span class="dim">pi5 bench</span>    54 ms inference     <span class="g">16.6 Hz</span> ✓
<span class="dim">pi5+camera</span>   66 ms end-to-end    <span class="g">15.1 Hz</span> ✓
<span class="dim">field</span>        live Bengal @3m     <span class="g">conf 0.63</span> ✓
<span class="dim">target</span>       ≥1 Hz               <span class="c">15× margin</span></div>
    </div>
  </div>
</section>

<!-- 5 · PI PORT -->
<section class="slide" data-state="PORT" data-color="#ffb454">
  <div class="kicker">// 04 · THE RASPBERRY PI PORT</div>
  <h2>Dependency <span class="accent">minefield</span>, cleared</h2>
  <div class="columns">
    <div>
      <ul class="sys">
        <li><b>cv2 was corrupted on arrival</b> <span class="d">— ImportError naming a library called “y”. ldd showed garbage ELF entries. Cause: damaged binary + dueling opencv-python/contrib installs.</span></li>
        <li><b>The pin matrix</b> <span class="d">— onnxruntime 1.18.1 needs numpy&lt;2; opencv 5.x drags numpy 2.4 in. Landed: numpy 1.26.4 + opencv-headless 4.11 + ort 1.18.1.</span></li>
        <li><b>Vendor API archaeology</b> <span class="d">— LED palette indices (red=0 green=1 blue=2 yellow=3), Ctrl_BEEP_Switch, Ctrl_Muto's signed-speed handling: all verified by on-device introspection.</span></li>
        <li><b>IR register idles at 0xFF</b> <span class="d">— “any key kills” fired instantly at startup until 255 was treated as no-key.</span></li>
      </ul>
    </div>
    <div>
      <div class="term-title">PI5 BENCHMARK · CATCHASER.BENCHMARK</div>
      <div class="term">
<span class="w">mode                inference    e2e     rate</span>
synthetic frame     54.1 ms    60.3 ms  <span class="g">16.6 Hz</span>
camera in loop      57.2 ms    66.2 ms  <span class="g">15.1 Hz</span>
--threads 4         56.1 ms      —      15.9 Hz

<span class="dim"># auto-threading already optimal on the Pi 5</span>
<span class="dim"># x86 → Pi5 scaling factor: ×2.6</span></div>
    </div>
  </div>
</section>

<!-- 6 · MOTION CONTROL -->
<section class="slide" data-state="PID" data-color="#5ee6d0">
  <div class="kicker">// 05 · MOTION CONTROL</div>
  <h2>Mecanum math + a PID <span class="accent">tamed by telemetry</span></h2>
  <div class="columns">
    <div>
      <div class="term-title">CATCHASER/CHASE.PY · THE MIX</div>
      <div class="term">
<span class="dim"># wheel mix — forward, clockwise turn, strafe</span>
l1 = forward + strafe + turn   <span class="dim"># front-left</span>
l2 = forward - strafe + turn   <span class="dim"># rear-left</span>
r1 = forward - strafe - turn   <span class="dim"># front-right</span>
r2 = forward + strafe - turn   <span class="dim"># rear-right</span>

<span class="dim"># heading error, normalized to half-frame</span>
err = (cat.cx - W/2) / (W/2)
turn = KP*err + KD*d_err       <span class="dim"># KP 80→40 after field data</span>

<span class="dim"># approach taper: gentle final 400mm</span>
fwd *= clamp((us_mm - 200)/400, 0.35, 1.0)</div>
    </div>
    <div>
      <p class="lede">The sim said KP=80. The carpet disagreed: at the real ~6 Hz loop one
      correction frame swung the nose <i>past</i> the cat — telemetry showed the error
      sign flipping in a single frame. Halved the gain, killed a D-term kick on fresh
      acquisitions, capped every wheel at 120.</p>
      <div class="term" style="margin-top:1rem">
<span class="dim">f0582</span> <span class="g">TRACKING</span> err=<span class="a">+0.38</span> turn=+44.4
<span class="dim">f0586</span> <span class="g">TRACKING</span> err=<span class="r">−0.27</span> turn=−31.8  <span class="dim">← overshoot!</span>
<span class="dim">f0588</span> <span class="a">LOST</span>     …and the cat is gone</div>
    </div>
  </div>
</section>

<!-- 7 · STATE MACHINE -->
<section class="slide" data-state="FSM" data-color="#6cff5c">
  <div class="kicker">// 06 · BEHAVIOR STATE MACHINE · LED = MOOD RING</div>
  <h2>Four states, <span class="accent">visible from the couch</span></h2>
  <div class="states">
    <div class="state-card"><h3><span class="led" style="background:#3ddc55;box-shadow:0 0 10px #3ddc55"></span>TRACKING</h3>
      <p>Cat in frame. Steer to center, drive forward — full speed when centered,
      turn-in-place past |err| 0.5. Deadband ±0.06 stops the jitters.</p></div>
    <div class="state-card"><h3><span class="led" style="background:#ffd23f;box-shadow:0 0 10px #ffd23f"></span>LOST</h3>
      <p>Cat vanished. Hold still 3 grace frames — a dropped detection shouldn't
      trigger a panic spin.</p></div>
    <div class="state-card"><h3><span class="led" style="background:#ff5470;box-shadow:0 0 10px #ff5470"></span>SEARCHING</h3>
      <p>Spin-and-stare: 2 frames rotating toward last-seen side, 3 frames parked
      for sharp detection frames. Continuous spinning outran the detector — motion
      blur made the cat invisible <i>while looking straight at it</i>.</p></div>
    <div class="state-card"><h3><span class="led" style="background:#5ee6d0;box-shadow:0 0 10px #5ee6d0"></span>HOLD</h3>
      <p>Ultrasonic &lt; 200 mm: stop advancing, keep pointing at the cat.
      The no-boop guarantee — the robot corners, never rams.</p></div>
  </div>
  <p class="lede" style="margin-top:1.4rem">Kill switches: any IR-remote key · Ctrl+C · software callback ·
  hard runtime limit. Motors stop in a <b style="color:#eaffe6">finally</b> block no matter how the loop dies.</p>
</section>

<!-- 8 · SIM -->
<section class="slide" data-state="SIM" data-color="#6ea8ff">
  <div class="kicker">// 07 · SIMULATION FIRST</div>
  <h2>PyBullet twin: <span class="accent">crash here, not there</span></h2>
  <div class="columns">
    <div>
      <p class="lede">The repo's SLAM simulator was repurposed as a chase test rig: same
      controller, simulated camera/ultrasonic/motors, plus a physical “cat” box the
      ray-cast ultrasonic genuinely sees. A synthetic detector projects the cat's true
      bearing through the camera model — validating the <i>control loop</i>, the part that was new.</p>
      <ul class="sys" style="margin-top:1rem">
        <li><b>approach</b> <span class="d">— acquire → center → drive → HOLD @238mm, no collision ✓</span></li>
        <li><b>search</b> <span class="d">— out-of-frame cat → grace → spin → acquire → HOLD @231mm ✓</span></li>
      </ul>
    </div>
    <div>
      <div class="term-title">TEST LEDGER</div>
      <div class="term">
<span class="g">239 passed</span>, 2 xfailed <span class="dim">(pre-existing, documented)</span>

chase controller unit tests        <span class="g">29</span>
dataset harvester                  <span class="g">8</span>
label review tool                  <span class="g">10</span>
pybullet end-to-end scenarios      <span class="g">2</span>
slam suite (inherited)             <span class="g">190</span>

<span class="dim"># first full-suite completion ever —</span>
<span class="dim"># an IMU deadlock had hung it forever</span></div>
    </div>
  </div>
</section>

<!-- 9 · FIELD RESULTS -->
<section class="slide" data-state="FIELD" data-color="#6cff5c">
  <div class="kicker">// 08 · FIELD RESULTS · LIVE BENGAL</div>
  <h2>Run 2 ended in HOLD <span class="accent">at the actual cat</span></h2>
  <div class="photo-grid">
    <figure class="hud"><img src="{{IMG_001}}" alt="Live cat detected at frame edge">
      <div class="corners"><i></i><i></i><i></i><i></i></div>
      <figcaption><span>LIVE CAT · EDGE ACQUISITION</span><span class="conf">0.63</span></figcaption></figure>
    <figure class="hud"><img src="{{IMG_012}}" alt="Close approach">
      <div class="corners"><i></i><i></i><i></i><i></i></div>
      <figcaption><span>FINAL APPROACH</span><span class="conf">0.53</span></figcaption></figure>
    <figure class="hud"><img src="{{IMG_013}}" alt="Point blank fur">
      <div class="corners"><i></i><i></i><i></i><i></i></div>
      <figcaption><span>POINT BLANK · MISSION COMPLETE</span><span class="conf">0.61</span></figcaption></figure>
  </div>
  <p class="lede" style="margin-top:1.3rem">Two 120 s untethered WiFi runs. Detection snapshots
  (auto-saved, annotated) turned debugging from guesswork into evidence — every claim in this
  deck traces to a frame or a log line.</p>
</section>

<!-- VIDEO_SLIDE_START -->
<!-- 9b · CHASE TAPE -->
<section class="slide" data-state="● REC" data-color="#ff5470">
  <div class="kicker">// 08b · the chase · uncut · <b>iPhone tape</b></div>
  <h2>Roll the <span class="accent">footage</span></h2>
  <div style="display:flex;gap:2.5rem;align-items:center;flex-wrap:wrap">
    <figure class="hud" style="flex:0 0 auto">
      <video src="{{VIDEO}}" poster="" controls playsinline preload="metadata"
             style="display:block;height:52vh;max-height:440px;width:auto;background:#000"></video>
      <div class="corners"><i></i><i></i><i></i><i></i></div>
      <figcaption><span>IMG_3375 · LIVING ROOM · 62 s</span><span class="conf">▶ TAP TO PLAY</span></figcaption>
    </figure>
    <div style="flex:1;min-width:260px">
      <p class="lede">The Cat Chaser 3000, in the field, in motion — the loop from
      the slides running for real: acquire, center, approach, and the cat's
      considered response.</p>
      <div class="term" style="margin-top:1.2rem">
<span class="dim">source</span>   IMG_3375.MOV · 1080p · 112 MB
<span class="dim">encoded</span>  720p H.264 · <span class="g">2.6 MB</span> · embedded here
<span class="dim">verdict</span>  cat: <span class="a">unbothered</span></div>
    </div>
  </div>
</section>

<!-- 10 · BLOOPERS -->
<section class="slide" data-state="FP" data-color="#ff5470">
  <div class="kicker">// 09 · THE BLOOPER REEL · WHAT THE ROBOT THOUGHT WAS A CAT</div>
  <h2>Failure analysis, <span class="accent">with receipts</span></h2>
  <div class="photo-grid">
    <figure class="hud"><img src="{{IMG_010}}" alt="Motion blurred trash can">
      <div class="corners"><i></i><i></i><i></i><i></i></div>
      <figcaption><span>TRASH CAN + MOTION BLUR</span><span class="conf bad">0.41 ✗</span></figcaption></figure>
    <figure class="hud"><img src="{{IMG_011}}" alt="A jacket on the floor">
      <div class="corners"><i></i><i></i><i></i><i></i></div>
      <figcaption><span>CRUMPLED JACKET</span><span class="conf bad">0.63 ✗</span></figcaption></figure>
    <figure class="hud"><img src="{{IMG_008}}" alt="iPad decoy">
      <div class="corners"><i></i><i></i><i></i><i></i></div>
      <figcaption><span>IPAD DECOY (intentional)</span><span class="conf">0.62 ✓</span></figcaption></figure>
  </div>
  <p class="lede" style="margin-top:1.3rem">Lessons burned in: rotation blur is the enemy (spin-and-stare),
  dim screens defeat decoys (max brightness), and every threshold in the config now cites the
  field frame that set it. Also: the “unreachable robot” was a stale WiFi profile broadcasting a
  dead address — trust nmcli, not the OLED.</p>
</section>

<!-- 11 · DATASET -->
<section class="slide" data-state="DATA" data-color="#5ee6d0">
  <div class="kicker">// 10 · DATASET PIPELINE · COUCH-DRIVEN ML-OPS</div>
  <h2>Chase → harvest → <span class="accent">thumb-triage</span></h2>
  <div class="columns">
    <div>
      <div class="term-title">PHONE WORKFLOW</div>
      <div class="term">
$ <span class="w">./harvest.sh</span>          <span class="dim"># chase + collect (≤1/s dets,</span>
                        <span class="dim">#  1/10s negatives, cap 300)</span>
$ <span class="w">./review.sh</span>           <span class="dim"># triage server :5000</span>

<span class="g">g</span> = Good   → keep detector's box
<span class="r">n</span> = No cat → force clean negative
<span class="a">f</span> = Fix    → route to real box editor
<span class="c">u</span> = Undo   → lossless, resume anytime</div>
    </div>
    <div>
      <div class="stat-row">
        <div><div class="big-num">244</div><div class="stat-label">frames triaged · 2 sessions</div></div>
        <div><div class="big-num">100<small>%</small></div><div class="stat-label">precision · 0 FP</div></div>
        <div><div class="big-num">90<small>%</small></div><div class="stat-label">recall · 23 misses</div></div>
      </div>
      <p class="lede">Triaging by thumb also <i>measures</i> the detector — every g/n/f tap is a
      confusion-matrix cell (next slide). It surfaced 23 missed cats hiding as "negatives"; the
      pipeline now flags them. Fine-tuning waits for ~500 varied positives — these sessions become
      the <i>eval set</i> any future model must beat.</p>
    </div>
  </div>
</section>

<!-- 11b · CONFUSION MATRIX -->
<section class="slide" data-state="EVAL" data-color="#6cff5c">
  <div class="kicker">// 10b · evaluation · <b>confusion matrix</b> · 244 reviewed frames</div>
  <h2>Precision <span class="accent">100%</span> · Recall <span class="accent">90%</span></h2>
  <div style="display:flex;gap:3.5rem;align-items:center;flex-wrap:wrap">
    <div class="cm">
      <div class="cm-corner"></div>
      <div class="cm-head">actual<br>CAT</div>
      <div class="cm-head">actual<br>NO CAT</div>
      <div class="cm-head">predicted<br>CAT</div>
      <div class="cm-cell tp"><b>198</b><span>true pos</span></div>
      <div class="cm-cell fp"><b>0</b><span>false pos</span></div>
      <div class="cm-head">predicted<br>NO CAT</div>
      <div class="cm-cell fn"><b>23</b><span>false neg</span></div>
      <div class="cm-cell tn"><b>23</b><span>true neg</span></div>
    </div>
    <div style="flex:1;min-width:280px">
      <div class="metric-row">
        <div class="metric"><b>100%</b><span>precision · 198/198</span></div>
        <div class="metric warn"><b>89.6%</b><span>recall · 198/221</span></div>
        <div class="metric"><b>94.5%</b><span>F1 score</span></div>
      </div>
      <p class="lede"><b style="color:var(--green)">Never cries cat falsely</b> (0 false
      positives) — the 0.50 threshold earns its keep. It <b style="color:var(--amber)">misses
      ~10%</b>, and the misses are all hard: motion blur, extreme close-ups, a dark cat under
      furniture, legs-only crops.</p>
      <p class="lede" style="margin-top:.8rem;font-size:.82rem">Ground truth from human triage
      of every frame. <b style="color:#eaffe6">Caveat:</b> negatives were sampled from
      detector-cleared frames, so recall is a strong estimate, not a held-out benchmark. Bonus
      finding: those 23 misses were silently mislabeled as negatives (a review-tool trap) — they
      must be re-boxed before any fine-tune, where they'd be the most valuable hard examples.</p>
    </div>
  </div>
</section>

<!-- 12 · ROADMAP -->
<section class="slide" data-state="NEXT" data-color="#ffb454">
  <div class="kicker">// 11 · ROADMAP</div>
  <h2>Next sorties <span class="accent">+ the SLAM campaign</span></h2>
  <div class="columns">
    <div>
      <ul class="sys">
        <li><b>Pan-servo search</b> <span class="d">— sweep the camera, not the chassis: sharp frames, 3× coverage, zero blur. One servo-sign check away.</span></li>
        <li><b>Exposure control</b> <span class="d">— shorter shutter to cut rotation blur at the source.</span></li>
        <li><b>Lead pursuit</b> <span class="d">— centroid velocity prediction, for when the Bengal stops going easy on us.</span></li>
        <li><b>IR kill verification</b> <span class="d">— the last unchecked box before fully unsupervised roaming.</span></li>
      </ul>
    </div>
    <div>
      <p class="lede">Separately: an external review confirmed the repo's SLAM cannot yet
      produce a metric map (VO counts frames, not metres; the EKF never sees a real
      camera measurement; the sim was anchored to ground truth). A six-phase plan is
      committed — honest sim first, keyframe VO, metric scale from ground-plane
      homography — decoupled from cat operations.</p>
      <div class="term" style="margin-top:1rem">
<span class="dim">phase 1</span>  honest measurement (de-anchor sim)
<span class="dim">phase 2</span>  keyframe VO
<span class="dim">phase 3</span>  metric scale        <span class="a">← the crux</span>
<span class="dim">phase 4</span>  real EKF updates
<span class="dim">phase 5</span>  IMU fusion
<span class="dim">phase 6</span>  record-replay + hardware ladder</div>
    </div>
  </div>
</section>

<!-- 13 · CREDITS -->
<section class="slide" data-state="EOF" data-color="#6cff5c">
  <div class="kicker">// 12 · END OF TRANSMISSION</div>
  <h2>Built in <span class="accent">two days</span>, debugged with evidence</h2>
  <div class="stat-row" style="margin-top:0">
    <div><div class="big-num">13<small>+</small></div><div class="stat-label">commits, cloud + local</div></div>
    <div><div class="big-num">239</div><div class="stat-label">tests green</div></div>
    <div><div class="big-num">15<small>Hz</small></div><div class="stat-label">detector on Pi 5</div></div>
    <div><div class="big-num">1</div><div class="stat-label">cat, cornered gently</div></div>
  </div>
  <ul class="sys" style="margin-top:1rem">
    <li><b>Stack:</b> <span class="d">Python · onnxruntime · OpenCV · PyBullet · Flask · one very patient Bengal</span></li>
    <li><b>Repo:</b> <span class="d">github.com/peter-m-mayer/rover — catchaser/ package, full history</span></li>
    <li><b>Method:</b> <span class="d">sim before hardware · telemetry before tuning · snapshots before speculation</span></li>
  </ul>
  <p class="title-sub" style="margin-top:2rem">The cat, when reached, was found to be
  <span style="color:var(--green)">unbothered</span>. Recommend continued observation. 🐾</p>
</section>

</div>

<button class="arrow left" onclick="nav(-1)">◀</button>
<button class="arrow right" onclick="nav(1)">▶</button>

<div class="hudbar">
  <span class="fcount" id="fcount">f0001 / f0013</span>
  <div class="dots" id="dots"></div>
  <span class="statechip" id="statechip">BOOT</span>
</div>

<script>
const slides=[...document.querySelectorAll('.slide')];
const dots=document.getElementById('dots');
slides.forEach((s,i)=>{const d=document.createElement('i');d.onclick=()=>go(i);dots.appendChild(d);});
let cur=0;
function go(i){
  cur=Math.max(0,Math.min(slides.length-1,i));
  slides.forEach((s,j)=>{s.classList.toggle('active',j===cur);s.classList.toggle('prev',j<cur);});
  [...dots.children].forEach((d,j)=>d.classList.toggle('on',j===cur));
  document.getElementById('fcount').textContent=`f${String(cur+1).padStart(4,'0')} / f${String(slides.length).padStart(4,'0')}`;
  const chip=document.getElementById('statechip');
  chip.textContent=slides[cur].dataset.state;
  chip.style.background=slides[cur].dataset.color;
  document.getElementById('tick').style.width=((cur+1)/slides.length*100)+'%';
  location.hash=cur+1;
}
function nav(d){go(cur+d)}
addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown')nav(1);
  if(e.key==='ArrowLeft'||e.key==='PageUp')nav(-1);
  if(e.key==='Home')go(0); if(e.key==='End')go(slides.length-1);
});
let tx=null;
addEventListener('touchstart',e=>tx=e.touches[0].clientX,{passive:true});
addEventListener('touchend',e=>{
  if(tx===null)return;
  const dx=e.changedTouches[0].clientX-tx;
  if(Math.abs(dx)>50)nav(dx<0?1:-1);
  tx=null;
},{passive:true});
go(parseInt(location.hash.slice(1))-1||0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/tmp/dets")
    ap.add_argument("--video", default=os.path.join(HERE, "catchaser_chase.mp4"))
    ap.add_argument("--out", default=os.path.join(HERE, "catchaser_presentation.html"))
    args = ap.parse_args()
    build(args.images, args.out, video_path=args.video)
