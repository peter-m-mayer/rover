"""Cat Chaser 3000 — rcbot: drive the robot from a web page.

A Flask app the robot serves so you can drive it from any browser (e.g. your
upstairs computer) with a live camera view and keyboard control. Mecanum wheels
give omnidirectional motion; held keys combine into one motion vector, so
up+left drives a diagonal, up+6 arcs, etc.

    python3 -m catchaser.rc          # on the robot; open http://<pi-ip>:5001

Keys (numpad-friendly, arrows too):
    8 / Up      forward           4   rotate left (CCW)     a / Left  slide left
    2 / Down    back              6   rotate right (CW)     s / Right slide right
    space / 5   STOP              -/+ slower / faster
    j  pan left     k  pan right     i  tilt up    m  tilt down
    h  camera home  0  set current camera pose as home (recalibrate)

Safety: releasing all keys stops the wheels; a server-side dead-man stops them
if the browser goes quiet; closing the tab stops the robot.
"""

import argparse
import sys
import threading
import time

from raspbot_slam import config

HOME_FILE = "~/.catchaser_rc_home.json"


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FrameGrabber:
    """Continuously grab the newest camera frame in a background thread.

    Decouples capture from the MJPEG stream so a transient camera read error
    can't kill the stream (it just reuses the last good frame and retries), and
    so multiple / reconnecting browser tabs all read one shared buffer instead
    of fighting over the single VideoCapture. Reopens the camera on repeated
    failure.
    """

    def __init__(self, camera, fps: float = 14.0):
        self._camera = camera
        self._period = 1.0 / fps
        self._latest = None
        self._lock = threading.Lock()
        self._stop = False
        self._fails = 0

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self._stop:
            try:
                frame = self._camera.capture_color()
                if frame is not None:
                    with self._lock:
                        self._latest = frame
                    self._fails = 0
            except Exception:
                self._fails += 1
                if self._fails >= 3:        # force a reopen on the next capture
                    try:
                        self._camera.close()
                    except Exception:
                        pass
                    self._fails = 0
                time.sleep(0.2)
            time.sleep(self._period)

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop = True


class RCController:
    """Manual drive + camera control. Pure of I/O timing — unit-testable."""

    def __init__(self, actuators, sensors=None, speed: int = 90,
                 pan: float = None, tilt: float = None):
        self.act = actuators
        self.sensors = sensors
        self.speed = int(speed)
        self.pan = float(config.SERVO_PAN_CENTER if pan is None else pan)
        self.tilt = float(config.SERVO_TILT_REST if tilt is None else tilt)
        self.home_pan = self.pan
        self.home_tilt = self.tilt

    # ---- wheels -------------------------------------------------------------
    def drive(self, forward: float, strafe: float, turn: float):
        """Drive from normalized [-1,1] inputs, scaled by speed and capped.

        + forward = ahead, + strafe = right, + turn = clockwise/right.
        """
        s = self.speed
        f, st, t = forward * s, strafe * s, turn * s
        peak = abs(f) + abs(st) + abs(t)
        cap = config.CHASE_MAX_SPEED
        if peak > cap and peak > 0:
            k = cap / peak
            f, st, t = f * k, st * k, t * k
        self.act.drive(f, t, st)
        return (f, t, st)

    def stop(self):
        self.act.stop()

    def set_speed(self, value: int):
        self.speed = int(_clamp(value, 20, config.MOTOR_SPEED_MAX))
        return self.speed

    def nudge_speed(self, delta: int):
        return self.set_speed(self.speed + delta)

    # ---- camera servos ------------------------------------------------------
    def pan_by(self, delta: float):
        self.pan = _clamp(self.pan + delta, config.SERVO_PAN_MIN, config.SERVO_PAN_MAX)
        self.act.set_servo_pan(self.pan)
        return self.pan

    def tilt_by(self, delta: float):
        self.tilt = _clamp(self.tilt + delta, config.SERVO_TILT_MIN, config.SERVO_TILT_MAX)
        self.act.set_servo_tilt(self.tilt)
        return self.tilt

    def go_home(self):
        self.pan, self.tilt = self.home_pan, self.home_tilt
        self.act.set_servo_pan(self.pan)
        self.act.set_servo_tilt(self.tilt)
        return (self.pan, self.tilt)

    def set_home(self):
        """Make the current camera pose the new home (recalibrate)."""
        self.home_pan, self.home_tilt = self.pan, self.tilt
        self._persist_home()
        return (self.home_pan, self.home_tilt)

    def distance_mm(self):
        if self.sensors is None:
            return -1
        try:
            return self.sensors.read_ultrasonic_mm()
        except Exception:
            return -1

    # ---- home persistence ---------------------------------------------------
    def _persist_home(self):
        import json
        import os
        try:
            with open(os.path.expanduser(HOME_FILE), "w") as f:
                json.dump({"pan": self.home_pan, "tilt": self.home_tilt}, f)
        except Exception:
            pass

    def load_home(self):
        import json
        import os
        p = os.path.expanduser(HOME_FILE)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    d = json.load(f)
                self.home_pan = float(d["pan"])
                self.home_tilt = float(d["tilt"])
            except Exception:
                pass
        return (self.home_pan, self.home_tilt)


_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,user-scalable=no">
<title>rcbot</title><style>
 :root{--g:#6cff5c;--bg:#0a0f0a;--panel:#121a12;--dim:#7d967a}
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 body{margin:0;background:var(--bg);color:#e6ffe6;font-family:ui-monospace,Menlo,Consolas,monospace;
   text-align:center;overflow:hidden}
 #view{width:100%;max-width:760px;background:#000;display:block;margin:0 auto;border-bottom:1px solid #234}
 .bar{padding:6px;font-size:.8rem;color:var(--dim);letter-spacing:.05em}
 .bar b{color:var(--g)}
 .pad{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;max-width:420px;margin:10px auto;padding:0 10px}
 button{font-family:inherit;font-size:1rem;padding:16px 6px;border:1px solid #2c3f2a;border-radius:12px;
   background:var(--panel);color:#e6ffe6;user-select:none}
 button:active,button.on{background:var(--g);color:#000;border-color:var(--g)}
 .stop{background:#3a0e14;border-color:#7a1f2b;color:#ff8ea0}
 .cam{background:#0e1622;border-color:#274}
 .row{display:flex;gap:8px;max-width:420px;margin:8px auto;padding:0 10px}
 .row button{flex:1}
 .hint{font-size:.68rem;color:#54644f;padding:6px 12px 20px;line-height:1.6}
</style></head><body>
<img id=view src="/stream.mjpg" alt="camera"
 onerror="setTimeout(()=>{view.src='/stream.mjpg?'+Date.now()},800)">
<div class=bar>speed <b id=spd>?</b> · pan <b id=pan>?</b> · tilt <b id=tlt>?</b>
  · dist <b id=dst>?</b>mm · <b id=drv>idle</b></div>
<div class=pad>
 <button class=cam data-k=i>tilt▲ (i)</button>
 <button data-k=8>fwd (8/↑)</button>
 <button class=cam data-k=0>set home (0)</button>
 <button data-k=4>⟲ rot (4)</button>
 <button class=stop data-k=" ">STOP (space)</button>
 <button data-k=6>rot ⟳ (6)</button>
 <button data-k=a>◀ slide (a)</button>
 <button data-k=2>back (2/↓)</button>
 <button data-k=s>slide ▶ (s)</button>
 <button class=cam data-k=j>pan◀ (j)</button>
 <button class=cam data-k=m>tilt▼ (m)</button>
 <button class=cam data-k=k>pan▶ (k)</button>
</div>
<div class=row>
 <button data-k=-> – slower</button>
 <button class=cam data-k=h>camera home (h)</button>
 <button data-k=+>faster +</button>
</div>
<div class=hint>hold to move · release = stop · arrows = drive+strafe · combine keys for diagonals
 · closing this tab stops the robot</div>
<script>
const held=new Set();
const MOVE=new Set(['8','2','4','6','a','s','ArrowUp','ArrowDown','ArrowLeft','ArrowRight',' ','5']);
function vec(){ // held movement keys -> normalized {forward,strafe,turn}
 let f=0,st=0,t=0;
 if(held.has('8')||held.has('ArrowUp'))f+=1;
 if(held.has('2')||held.has('ArrowDown'))f-=1;
 if(held.has('a')||held.has('ArrowLeft'))st-=1;
 if(held.has('s')||held.has('ArrowRight'))st+=1;
 if(held.has('4'))t-=1;      // rotate left / CCW
 if(held.has('6'))t+=1;      // rotate right / CW
 if(held.has(' ')||held.has('5')){f=0;st=0;t=0;}
 return {f,st,t};
}
async function post(url,body){try{return await(await fetch(url,{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})})).json();}catch(e){}}
let moving=false;
async function tick(){const v=vec();
 if(v.f||v.st||v.t){moving=true;const r=await post('/api/drive',{forward:v.f,strafe:v.st,turn:v.t});
   if(r)drv.textContent='drive';}
 else if(moving){moving=false;await post('/api/stop');drv.textContent='idle';}}
setInterval(tick,100);
async function servo(k){
 if(k==='i')show(await post('/api/servo',{tilt:+6}));
 else if(k==='m')show(await post('/api/servo',{tilt:-6}));
 else if(k==='j')show(await post('/api/servo',{pan:+6}));   // pan left
 else if(k==='k')show(await post('/api/servo',{pan:-6}));   // pan right
 else if(k==='h')show(await post('/api/home'));
 else if(k==='0')show(await post('/api/sethome'));
 else if(k==='+'||k==='=')show(await post('/api/speed',{delta:+10}));
 else if(k==='-')show(await post('/api/speed',{delta:-10}));
}
function show(s){if(!s)return; if(s.pan!=null)pan.textContent=Math.round(s.pan);
 if(s.tilt!=null)tlt.textContent=Math.round(s.tilt); if(s.speed!=null)spd.textContent=s.speed;}
function press(k){if(MOVE.has(k))held.add(k);else servo(k);
 document.querySelectorAll('[data-k]').forEach(b=>{if(b.dataset.k===k)b.classList.add('on');});}
function release(k){held.delete(k);
 document.querySelectorAll('[data-k]').forEach(b=>{if(b.dataset.k===k)b.classList.remove('on');});}
document.addEventListener('keydown',e=>{let k=e.key;if(k==='ArrowUp'||k==='ArrowDown'||
 k==='ArrowLeft'||k==='ArrowRight'||MOVE.has(k))e.preventDefault();press(k);});
document.addEventListener('keyup',e=>release(e.key));
// on-screen buttons: press-and-hold
document.querySelectorAll('[data-k]').forEach(b=>{const k=b.dataset.k;
 const dn=e=>{e.preventDefault();press(k);};const up=e=>{e.preventDefault();release(k);};
 b.addEventListener('mousedown',dn);b.addEventListener('mouseup',up);b.addEventListener('mouseleave',up);
 b.addEventListener('touchstart',dn,{passive:false});b.addEventListener('touchend',up,{passive:false});});
window.addEventListener('blur',()=>{held.clear();post('/api/stop');});
window.addEventListener('beforeunload',()=>{navigator.sendBeacon&&navigator.sendBeacon('/api/stop');});
async function poll(){const s=await(await fetch('/api/state')).json();show(s);
 if(s.distance!=null)dst.textContent=s.distance;}
setInterval(poll,500);poll();
</script></body></html>"""


def make_app(rc, frame_source=None):
    from flask import Flask, Response, request, jsonify
    app = Flask(__name__)
    app.config["_last_drive"] = [0.0]

    @app.route("/")
    def index():
        return _PAGE

    @app.route("/api/drive", methods=["POST"])
    def drive():
        b = request.get_json(force=True, silent=True) or {}
        app.config["_last_drive"][0] = time.monotonic()
        f, t, s = rc.drive(float(b.get("forward", 0)), float(b.get("strafe", 0)),
                           float(b.get("turn", 0)))
        return jsonify({"forward": f, "turn": t, "strafe": s})

    @app.route("/api/stop", methods=["POST"])
    def stop():
        rc.stop()
        return jsonify({"ok": True})

    @app.route("/api/servo", methods=["POST"])
    def servo():
        b = request.get_json(force=True, silent=True) or {}
        if "pan" in b:
            rc.pan_by(float(b["pan"]))
        if "tilt" in b:
            rc.tilt_by(float(b["tilt"]))
        return jsonify({"pan": rc.pan, "tilt": rc.tilt})

    @app.route("/api/home", methods=["POST"])
    def home():
        rc.go_home()
        return jsonify({"pan": rc.pan, "tilt": rc.tilt})

    @app.route("/api/sethome", methods=["POST"])
    def sethome():
        rc.set_home()
        return jsonify({"pan": rc.pan, "tilt": rc.tilt, "home": True})

    @app.route("/api/speed", methods=["POST"])
    def speed():
        b = request.get_json(force=True, silent=True) or {}
        if "delta" in b:
            rc.nudge_speed(int(b["delta"]))
        elif "value" in b:
            rc.set_speed(int(b["value"]))
        return jsonify({"speed": rc.speed})

    @app.route("/api/state")
    def state():
        return jsonify({"pan": rc.pan, "tilt": rc.tilt, "speed": rc.speed,
                        "distance": rc.distance_mm()})

    @app.route("/stream.mjpg")
    def stream():
        if frame_source is None:
            return Response(status=503)

        def gen():
            import cv2
            while True:
                frame = frame_source()          # newest buffered frame (or None)
                if frame is None:
                    time.sleep(0.05)
                    continue
                try:
                    ok, buf = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                except Exception:
                    time.sleep(0.05)
                    continue                    # never let one bad frame end the stream
                if ok:
                    yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
                time.sleep(0.07)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rcbot — web drive controller")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--speed", type=int, default=90)
    ap.add_argument("--deadman", type=float, default=0.6,
                    help="stop the wheels if no drive command for this long (s)")
    args = ap.parse_args(argv)

    from .hw import make_hardware
    bot, camera, sensors, actuators = make_hardware()
    if bot is None:
        print("[rcbot] vendor driver not found — run on the robot.", file=sys.stderr)
        return 2

    rc = RCController(actuators, sensors, speed=args.speed)
    rc.load_home()
    rc.go_home()
    grabber = FrameGrabber(camera).start()
    app = make_app(rc, frame_source=grabber.latest)

    # Dead-man watchdog: if the browser stops sending drive commands (closed
    # tab, dropped WiFi), stop the wheels.
    def watchdog():
        while True:
            time.sleep(0.2)
            last = app.config["_last_drive"][0]
            if last and (time.monotonic() - last) > args.deadman:
                app.config["_last_drive"][0] = 0.0
                try:
                    rc.stop()
                except Exception:
                    pass
    threading.Thread(target=watchdog, daemon=True).start()

    print(f"[rcbot] open http://<this-host>:{args.port}   (speed {args.speed}, "
          f"dead-man {args.deadman}s)")
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        actuators.stop()
        camera.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
