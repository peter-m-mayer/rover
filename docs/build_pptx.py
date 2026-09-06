#!/usr/bin/env python3
"""Build the Cat Chaser 3000 PPTX deck (dark engineering style, real photos).

Usage: python3 docs/build_pptx.py [--images /tmp/dets] [--out docs/catchaser_presentation.pptx]
"""

import argparse
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))

BG = RGBColor(0x07, 0x0B, 0x07)
PANEL = RGBColor(0x0D, 0x13, 0x0C)
GREEN = RGBColor(0x6C, 0xFF, 0x5C)
GREEN_DIM = RGBColor(0x3D, 0x9A, 0x45)
AMBER = RGBColor(0xFF, 0xB4, 0x54)
RED = RGBColor(0xFF, 0x54, 0x70)
CYAN = RGBColor(0x5E, 0xE6, 0xD0)
TEXT = RGBColor(0xC9, 0xDF, 0xC2)
TEXT_DIM = RGBColor(0x7D, 0x96, 0x7A)
WHITE = RGBColor(0xEA, 0xFF, 0xE6)

HEAD_FONT = "Chakra Petch"   # falls back gracefully if not installed
MONO_FONT = "Consolas"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def add_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    return slide


def text_box(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def para(tf, text, size=14, color=TEXT, bold=False, font=MONO_FONT,
         first=False, space_after=6, align=None):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    p.space_after = Pt(space_after)
    if align:
        p.alignment = align
    return p


def kicker(slide, text):
    tf = text_box(slide, Inches(0.7), Inches(0.35), Inches(12), Inches(0.4))
    para(tf, text.upper(), size=11, color=GREEN_DIM, first=True)


def title(slide, text, top=Inches(0.75), size=34):
    tf = text_box(slide, Inches(0.7), top, Inches(12), Inches(0.9))
    para(tf, text, size=size, color=WHITE, bold=True, font=HEAD_FONT, first=True)


def bullets(slide, items, left, top, width, size=13):
    tf = text_box(slide, left, top, width, Inches(5))
    for i, (head, body) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run(); r1.text = "▸ " + head + "  "
        r1.font.name = MONO_FONT; r1.font.size = Pt(size)
        r1.font.color.rgb = WHITE; r1.font.bold = True
        r2 = p.add_run(); r2.text = body
        r2.font.name = MONO_FONT; r2.font.size = Pt(size)
        r2.font.color.rgb = TEXT_DIM
        p.space_after = Pt(10)


def code_panel(slide, lines, left, top, width, height, size=11):
    shape = slide.shapes.add_shape(1, left, top, width, height)  # rectangle
    shape.fill.solid(); shape.fill.fore_color.rgb = PANEL
    shape.line.color.rgb = RGBColor(0x22, 0x33, 0x1F); shape.line.width = Pt(1)
    tf = shape.text_frame
    tf.word_wrap = False
    tf.margin_left = Emu(120000); tf.margin_top = Emu(90000)
    for i, (text, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run(); run.text = text
        run.font.name = MONO_FONT; run.font.size = Pt(size)
        run.font.color.rgb = color
        p.space_after = Pt(2)


def photo(slide, path, left, top, width, caption, conf, good=True):
    pic = slide.shapes.add_picture(path, left, top, width=width)
    cap_top = Emu(top + pic.height)
    tf = text_box(slide, left, cap_top, width, Inches(0.35))
    p = tf.paragraphs[0]
    r1 = p.add_run(); r1.text = caption + "  "
    r1.font.name = MONO_FONT; r1.font.size = Pt(10); r1.font.color.rgb = TEXT_DIM
    r2 = p.add_run(); r2.text = conf
    r2.font.name = MONO_FONT; r2.font.size = Pt(10); r2.font.bold = True
    r2.font.color.rgb = GREEN if good else RED


def stat(slide, left, top, num, label, color=GREEN):
    tf = text_box(slide, left, top, Inches(2.6), Inches(1.1))
    para(tf, num, size=32, color=color, bold=True, font=HEAD_FONT, first=True, space_after=0)
    para(tf, label.upper(), size=9, color=TEXT_DIM, space_after=0)


def build(images_dir, out_path, video_path=None, poster_path=None):
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    im = lambda k: os.path.join(images_dir, f"det_{k}.jpg")

    # ---- 1 · title ----------------------------------------------------------
    s = add_slide(prs)
    tf = text_box(s, Inches(0.8), Inches(1.9), Inches(11.5), Inches(2.6))
    para(tf, "CAT CHASER 3000", size=64, color=WHITE, bold=True, font=HEAD_FONT, first=True)
    para(tf, "An autonomous cat-pursuit robot — field report, 2026-09-05 → 07", size=16,
         color=GREEN, space_after=14)
    para(tf, "Yahboom RASPBOT-V2 · Raspberry Pi 5 · YOLOv8n/COCO · PID mecanum control "
             "· PyBullet-validated · one unimpressed Bengal", size=13, color=TEXT_DIM)
    tf2 = text_box(s, Inches(0.8), Inches(6.5), Inches(11.5), Inches(0.5))
    para(tf2, "PI 5 AARCH64   ·   DETECTOR 15 HZ   ·   239 TESTS GREEN   ·   CATS HARMED: 0",
         size=12, color=GREEN_DIM, first=True)

    # ---- 2 · mission architecture ------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 01 · mission architecture")
    title(s, "One loop, five links, zero mercy")
    code_panel(s, [
        ("camera 640x480 BGR", WHITE),
        ("   |", TEXT_DIM),
        ("YOLOv8n ONNX @320px  ->  cat boxes (COCO class 15), conf >= 0.50", GREEN),
        ("   |", TEXT_DIM),
        ("centroid -> heading error     err = (cx - W/2)/(W/2)  in [-1, +1]", WHITE),
        ("   |", TEXT_DIM),
        ("PID (KP=40 KI=0 KD=6, hardware-calibrated)  ->  turn", GREEN),
        ("   |", TEXT_DIM),
        ("mecanum mix  drive(forward, turn)  ->  4 wheel speeds, cap 120", WHITE),
        ("   |", TEXT_DIM),
        ("safety: ultrasonic HOLD @200mm · IR kill · Ctrl+C · runtime limit", AMBER),
    ], Inches(0.7), Inches(1.9), Inches(11.9), Inches(3.6), size=13)
    tf = text_box(s, Inches(0.7), Inches(5.8), Inches(11.9), Inches(1))
    para(tf, "Same controller runs on real hardware and the PyBullet twin — the sim caught "
             "the design bugs, the carpet caught the physics bugs.", size=13, color=TEXT_DIM, first=True)

    # ---- 3 · platform -------------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 02 · the platform")
    title(s, "Yahboom RASPBOT-V2 + Raspberry Pi 5")
    bullets(s, [
        ("Compute", "Pi 5 Model B rev 1.1 · aarch64 · Python 3.11"),
        ("MCU bridge", "I2C bus 1 @ 0x2B — motors, servos, LEDs, buzzer"),
        ("Drive", "4x mecanum wheels, ±255, no encoders — \"motor commands are suggestions\""),
        ("Camera", "USB 640x480, fixed focus (~0.4 m to infinity)"),
        ("Ranging", "forward ultrasonic, mm resolution"),
        ("Status", "14x WS2812B LEDs + buzzer + IR remote (kill switch)"),
    ], Inches(0.7), Inches(1.9), Inches(6.6))
    code_panel(s, [
        ("vendor API archaeology (on-device):", GREEN_DIM),
        ("", TEXT),
        ("docs promised:   Ctrl_WS2812B(id, r, g, b)", RED),
        ("hardware ships:  Ctrl_WQ2812_ALL(state, idx)", GREEN),
        ("                 red=0 green=1 blue=2 yellow=3", TEXT_DIM),
        ("buzzer:          Ctrl_BEEP_Switch(state)", GREEN),
        ("IR idle byte:    0xFF  (not 0 — kill switch", GREEN),
        ("                 fired instantly until fixed)", TEXT_DIM),
    ], Inches(7.6), Inches(1.9), Inches(5.1), Inches(2.9), size=11)

    # ---- 4 · neural net -----------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 03 · the neural net · provenance")
    title(s, "YOLOv8-nano, COCO-raised, cat-filtered")
    bullets(s, [
        ("Origin", "Ultralytics YOLOv8n, pretrained on COCO 2017 (118k images, 80 classes). "
                   "We keep exactly one: class 15, cat."),
        ("Export", "yolov8n.pt -> ONNX · 320px input · opset 12 · simplified · 12.1 MB · ~3.2M params"),
        ("Runtime", "pure onnxruntime + OpenCV on the robot — no ultralytics, no torch. "
                    "Letterbox -> NCHW -> NMS in ~150 lines."),
        ("Threshold", "conf >= 0.50, field-calibrated: real cat always >= 0.53, false positives 0.41–0.45"),
        ("Fine-tune?", "Not yet — 99.5% precision over 244 triaged frames, 0 false positives. Pipeline ready."),
    ], Inches(0.7), Inches(1.9), Inches(6.7))
    code_panel(s, [
        ("VALIDATION LADDER", GREEN_DIM),
        ("", TEXT),
        ("x86 sanity    real-cat photo    conf 0.94  ✓", TEXT),
        ("x86 bench     21 ms/frame       47 Hz      ✓", TEXT),
        ("pi5 bench     54 ms inference   16.6 Hz    ✓", GREEN),
        ("pi5+camera    66 ms end-to-end  15.1 Hz    ✓", GREEN),
        ("field         live Bengal @3m   conf 0.63  ✓", GREEN),
        ("target        >= 1 Hz           15x margin", CYAN),
    ], Inches(7.7), Inches(1.9), Inches(5.0), Inches(2.9), size=11)

    # ---- 5 · pi port --------------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 04 · the raspberry pi port")
    title(s, "Dependency minefield, cleared")
    bullets(s, [
        ("cv2 corrupted on arrival", "ImportError naming a library called 'y'; ldd showed garbage "
         "ELF entries. Damaged binary + dueling opencv installs."),
        ("The pin matrix", "onnxruntime 1.18.1 needs numpy<2; opencv 5.x drags in numpy 2.4. "
         "Landed: numpy 1.26.4 + opencv-headless 4.11 + ort 1.18.1."),
        ("64-bit dividend", "aarch64 confirmed on-device -> onnxruntime wheels just work; "
         "the feared Pi-3/TFLite fallback never happened."),
    ], Inches(0.7), Inches(1.9), Inches(6.6))
    code_panel(s, [
        ("PI 5 BENCHMARK · catchaser.benchmark", GREEN_DIM),
        ("", TEXT),
        ("mode              inference   e2e      rate", WHITE),
        ("synthetic frame   54.1 ms     60.3 ms  16.6 Hz", TEXT),
        ("camera in loop    57.2 ms     66.2 ms  15.1 Hz", GREEN),
        ("--threads 4       56.1 ms     —        15.9 Hz", TEXT),
        ("", TEXT),
        ("# auto-threading already optimal", TEXT_DIM),
        ("# x86 -> pi5 scaling factor: x2.6", TEXT_DIM),
    ], Inches(7.6), Inches(1.9), Inches(5.1), Inches(2.9), size=11)

    # ---- 6 · motion control -------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 05 · motion control")
    title(s, "Mecanum math + a PID tamed by telemetry")
    code_panel(s, [
        ("# wheel mix — forward, clockwise turn, strafe", TEXT_DIM),
        ("l1 = forward + strafe + turn    # front-left", WHITE),
        ("l2 = forward - strafe + turn    # rear-left", WHITE),
        ("r1 = forward - strafe - turn    # front-right", WHITE),
        ("r2 = forward + strafe - turn    # rear-right", WHITE),
        ("", TEXT),
        ("err  = (cat.cx - W/2) / (W/2)   # normalized", GREEN),
        ("turn = KP*err + KD*d_err        # KP 80 -> 40", GREEN),
        ("fwd *= clamp((us-200)/400, .35, 1)  # taper", GREEN),
    ], Inches(0.7), Inches(1.9), Inches(6.2), Inches(3.2), size=12)
    code_panel(s, [
        ("THE OVERSHOOT, CAUGHT IN TELEMETRY", GREEN_DIM),
        ("", TEXT),
        ("f0582 TRACKING err=+0.38 turn=+44.4", TEXT),
        ("f0586 TRACKING err=-0.27 turn=-31.8  <- flipped!", AMBER),
        ("f0588 LOST     ...and the cat is gone", RED),
        ("", TEXT),
        ("sim said KP=80; the carpet said KP=40.", TEXT_DIM),
        ("plus: D-term now needs 2 tracked frames", TEXT_DIM),
        ("(fresh-acquisition kick was a bug)", TEXT_DIM),
    ], Inches(7.2), Inches(1.9), Inches(5.5), Inches(3.2), size=11)

    # ---- 7 · state machine --------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 06 · behavior state machine · led = mood ring")
    title(s, "Four states, visible from the couch")
    cards = [
        ("TRACKING", GREEN, "Cat in frame: steer to center, drive forward. Full speed centered; "
         "turn-in-place past |err| 0.5; deadband ±0.06."),
        ("LOST", AMBER, "Cat vanished: hold 3 grace frames — a dropped detection shouldn't "
         "trigger a panic spin."),
        ("SEARCHING", RED, "Spin-and-stare: 2 frames rotating toward last-seen side, 3 parked "
         "for sharp frames. Continuous spin outran the detector."),
        ("HOLD", CYAN, "Ultrasonic < 200mm: stop advancing, keep aiming. The no-boop guarantee."),
    ]
    for i, (name, color, body) in enumerate(cards):
        left = Inches(0.7 + (i % 2) * 6.2)
        top = Inches(1.9 + (i // 2) * 2.1)
        shape = s.shapes.add_shape(1, left, top, Inches(5.9), Inches(1.85))
        shape.fill.solid(); shape.fill.fore_color.rgb = PANEL
        shape.line.color.rgb = color; shape.line.width = Pt(1.25)
        tf = shape.text_frame; tf.word_wrap = True
        tf.margin_left = Emu(140000); tf.margin_top = Emu(90000)
        para(tf, "● " + name, size=15, color=color, bold=True, font=HEAD_FONT, first=True)
        para(tf, body, size=11, color=TEXT_DIM)
    tf = text_box(s, Inches(0.7), Inches(6.35), Inches(12), Inches(0.7))
    para(tf, "Kill switches: any IR-remote key · Ctrl+C · software callback · runtime limit. "
             "Motors stop in a finally block no matter how the loop dies.", size=12,
         color=TEXT_DIM, first=True)

    # ---- 8 · sim ------------------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 07 · simulation first")
    title(s, "PyBullet twin: crash here, not there")
    bullets(s, [
        ("Digital twin", "same controller, simulated camera/ultrasonic/motors; a physical 'cat' "
         "box the ray-cast ultrasonic genuinely sees."),
        ("Synthetic detector", "projects the cat's true bearing through the camera model — "
         "validates the control loop, the part that was new."),
        ("approach scenario", "acquire -> center -> drive -> HOLD @238mm, no collision ✓"),
        ("search scenario", "out-of-frame cat -> grace -> spin -> acquire -> HOLD @231mm ✓"),
    ], Inches(0.7), Inches(1.9), Inches(6.6))
    code_panel(s, [
        ("TEST LEDGER", GREEN_DIM),
        ("", TEXT),
        ("239 passed, 2 xfailed (documented)", GREEN),
        ("", TEXT),
        ("chase controller units          29", TEXT),
        ("dataset harvester                8", TEXT),
        ("label review tool               10", TEXT),
        ("pybullet end-to-end              2", TEXT),
        ("slam suite (inherited)         190", TEXT),
        ("", TEXT),
        ("# first full-suite completion ever —", TEXT_DIM),
        ("# an IMU deadlock had hung it forever", TEXT_DIM),
    ], Inches(7.7), Inches(1.9), Inches(5.0), Inches(3.4), size=11)

    # ---- 9 · field results (photos) ----------------------------------------
    s = add_slide(prs)
    kicker(s, "// 08 · field results · live bengal")
    title(s, "Run 2 ended in HOLD at the actual cat")
    w = Inches(3.9)
    photo(s, im("001"), Inches(0.7), Inches(2.0), w, "LIVE CAT · EDGE ACQUISITION", "0.63")
    photo(s, im("012"), Inches(4.75), Inches(2.0), w, "FINAL APPROACH", "0.53")
    photo(s, im("013"), Inches(8.8), Inches(2.0), w, "POINT BLANK · COMPLETE", "0.61")
    tf = text_box(s, Inches(0.7), Inches(5.7), Inches(12), Inches(1.2))
    para(tf, "Two 120 s untethered WiFi runs. Auto-saved detection snapshots turned debugging "
             "from guesswork into evidence — every claim in this deck traces to a frame or a "
             "log line.", size=12, color=TEXT_DIM, first=True)

    # ---- 9b · chase tape (embedded video) ----------------------------------
    if video_path and os.path.exists(video_path):
        s = add_slide(prs)
        kicker(s, "// 08b · the chase · uncut · iphone tape")
        title(s, "Roll the footage")
        # Portrait video: place it left, tall; notes on the right.
        vid_w, vid_h = Inches(3.1), Inches(5.5)
        vid_left, vid_top = Inches(0.9), Inches(1.7)
        try:
            s.shapes.add_movie(
                video_path, vid_left, vid_top, vid_w, vid_h,
                poster_frame_image=(poster_path if poster_path
                                    and os.path.exists(poster_path) else None),
                mime_type="video/mp4")
        except Exception as e:
            tf = text_box(s, vid_left, vid_top, vid_w, Inches(1))
            para(tf, f"[video embed failed: {e}]", size=11, color=RED, first=True)
        code_panel(s, [
            ("THE CHASE · IMG_3375", GREEN_DIM),
            ("", TEXT),
            ("source    1080p · 112 MB", TEXT),
            ("encoded   720p H.264 · 2.6 MB", GREEN),
            ("embedded  right here in this deck", TEXT),
            ("", TEXT),
            ("the loop from the slides, running", TEXT_DIM),
            ("for real: acquire, center, approach,", TEXT_DIM),
            ("and the cat's considered response.", TEXT_DIM),
            ("", TEXT),
            ("verdict   cat: unbothered", AMBER),
        ], Inches(4.6), Inches(1.9), Inches(8.0), Inches(3.6), size=13)

    # ---- 10 · bloopers ------------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 09 · the blooper reel · what the robot thought was a cat")
    title(s, "Failure analysis, with receipts")
    photo(s, im("010"), Inches(0.7), Inches(2.0), w, "TRASH CAN + MOTION BLUR", "0.41 ✗", good=False)
    photo(s, im("011"), Inches(4.75), Inches(2.0), w, "CRUMPLED JACKET", "0.63 ✗", good=False)
    photo(s, im("008"), Inches(8.8), Inches(2.0), w, "IPAD DECOY (intentional)", "0.62 ✓")
    tf = text_box(s, Inches(0.7), Inches(5.7), Inches(12), Inches(1.3))
    para(tf, "Lessons burned in: rotation blur is the enemy (spin-and-stare) · dim screens defeat "
             "decoys (max brightness) · every config threshold now cites the field frame that set "
             "it · and the \"unreachable robot\" was a stale WiFi profile broadcasting a dead "
             "address — trust nmcli, not the OLED.", size=12, color=TEXT_DIM, first=True)

    # ---- 11 · dataset -------------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 10 · dataset pipeline · couch-driven ml-ops")
    title(s, "Chase -> harvest -> thumb-triage")
    code_panel(s, [
        ("PHONE WORKFLOW", GREEN_DIM),
        ("", TEXT),
        ("$ ./harvest.sh    # chase + collect (<=1/s dets,", WHITE),
        ("                  #  1/10s negatives, cap 300)", TEXT_DIM),
        ("$ ./review.sh     # triage web server :5000", WHITE),
        ("", TEXT),
        ("g = Good    keep detector's box", GREEN),
        ("n = No cat  force clean negative", RED),
        ("f = Fix     route to real box editor", AMBER),
        ("u = Undo    lossless, resume anytime", CYAN),
    ], Inches(0.7), Inches(1.9), Inches(6.0), Inches(3.4), size=12)
    stat(s, Inches(7.3), Inches(2.0), "244", "frames triaged · 2 sessions")
    stat(s, Inches(9.9), Inches(2.0), "99.5%", "detector precision")
    stat(s, Inches(7.3), Inches(3.3), "198/46", "positives / negatives")
    stat(s, Inches(9.9), Inches(3.3), "0 FP", "false positives · 1 box-fix", color=CYAN)
    tf = text_box(s, Inches(7.3), Inches(4.6), Inches(5.3), Inches(2))
    para(tf, "Of 198 detector cat-claims: 197 confirmed, 1 box-fix, zero false positives. "
             "Fine-tuning waits for ~500 varied positives (40% there); these sessions become "
             "the eval set any future model must beat.", size=12, color=TEXT_DIM, first=True)

    # ---- 12 · roadmap -------------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 11 · roadmap")
    title(s, "Next sorties + the SLAM campaign")
    bullets(s, [
        ("Pan-servo search", "sweep the camera, not the chassis: sharp frames, 3x coverage, "
         "zero blur. One servo-sign check away."),
        ("Exposure control", "shorter shutter to cut rotation blur at the source."),
        ("Lead pursuit", "centroid velocity prediction, for when the Bengal stops going easy."),
        ("IR kill verification", "the last unchecked box before unsupervised roaming."),
    ], Inches(0.7), Inches(1.9), Inches(6.4))
    code_panel(s, [
        ("SLAM CAMPAIGN (separate track)", GREEN_DIM),
        ("", TEXT),
        ("external review: no metric map possible yet", TEXT_DIM),
        ("(VO counts frames not metres; EKF never", TEXT_DIM),
        (" sees a real camera measurement)", TEXT_DIM),
        ("", TEXT),
        ("phase 1  honest measurement (de-anchor sim)", TEXT),
        ("phase 2  keyframe VO", TEXT),
        ("phase 3  metric scale        <- the crux", AMBER),
        ("phase 4  real EKF updates", TEXT),
        ("phase 5  IMU fusion", TEXT),
        ("phase 6  record-replay + hardware ladder", TEXT),
    ], Inches(7.4), Inches(1.9), Inches(5.3), Inches(3.6), size=11)

    # ---- 13 · credits -------------------------------------------------------
    s = add_slide(prs)
    kicker(s, "// 12 · end of transmission")
    title(s, "Built in two days, debugged with evidence")
    stat(s, Inches(0.8), Inches(2.1), "13+", "commits, cloud + local")
    stat(s, Inches(3.9), Inches(2.1), "239", "tests green")
    stat(s, Inches(7.0), Inches(2.1), "15 Hz", "detector on pi 5")
    stat(s, Inches(10.1), Inches(2.1), "1", "cat, cornered gently")
    bullets(s, [
        ("Stack", "Python · onnxruntime · OpenCV · PyBullet · Flask · one very patient Bengal"),
        ("Repo", "github.com/peter-m-mayer/rover — catchaser/ package, full history"),
        ("Method", "sim before hardware · telemetry before tuning · snapshots before speculation"),
    ], Inches(0.8), Inches(3.7), Inches(11.5))
    tf = text_box(s, Inches(0.8), Inches(6.1), Inches(11.5), Inches(0.8))
    para(tf, "The cat, when reached, was found to be unbothered. Recommend continued observation.",
         size=14, color=GREEN, first=True)

    prs.save(out_path)
    print(f"wrote {out_path} ({os.path.getsize(out_path)//1024} KB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/tmp/dets")
    ap.add_argument("--video", default=os.path.join(HERE, "catchaser_chase.mp4"))
    ap.add_argument("--poster", default=os.path.join(HERE, "chase_poster.jpg"))
    ap.add_argument("--out", default=os.path.join(HERE, "catchaser_presentation.pptx"))
    args = ap.parse_args()
    build(args.images, args.out, video_path=args.video, poster_path=args.poster)
