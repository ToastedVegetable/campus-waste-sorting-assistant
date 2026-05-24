"""
app.py  (LLM capture-on-demand variant)
========================================
A Tkinter app that shows a live webcam preview with a Capture button. When you
press Capture it freezes the current frame, sends it to a vision LLM ONCE, and
shows what the item is and which bin it belongs in.

Flow
----
  [Camera thread] continuously grabs frames for the live preview.
  [You] press "Capture & Classify".
  [Worker thread] sends a clean, unannotated frame to the LLM and waits for the
                  answer (so the UI never freezes during the network call).
  [UI] shows the frozen frame + result, and highlights the recommended bin.
  [You] press "Resume Live" (or Capture again) to go back to the preview.

Visuals
-------
  * Northwestern-purple theme.
  * A local preview-only highlight box is drawn on the live feed to help you aim.
    The box is only a UI overlay; it is never drawn onto the image sent to the
    LLM.
"""

import threading
import time
import os

import cv2
from PIL import Image, ImageTk
import tkinter as tk

from waste_sorter import config
from .llm_client import LLMClassifier


DISPLAY_MAX_W = 760
DISPLAY_MAX_H = 560

# ---- Northwestern-purple theme (official NU purple is #4E2A84) ----
NU_PURPLE = "#4E2A84"
THEME_BG = "#2e1b4d"          # deep purple background
THEME_PANEL = "#4E2A84"       # brand-purple panels / idle bin cards
THEME_TEXT = "#ffffff"
THEME_SUBTLE = "#cdbdec"      # light lavender for secondary text
BTN_PRIMARY = "#8a63d2"       # lighter purple -> readable white text
BTN_PRIMARY_ACTIVE = "#9d7ce0"
BTN_SECONDARY = "#5d3a93"
BTN_SECONDARY_ACTIVE = "#6e49a8"
GUIDE_COLOR = (210, 180, 255)  # light purple (RGB) for the guide box
ITEM_HIGHLIGHT_COLOR = (104, 255, 188)  # mint green (RGB), local UI only

# Guide box size as a fraction of the frame (centered). This is only used as a
# local search area for the preview highlight; it is not sent to Gemini.
ROI_W_FRAC = 0.66
ROI_H_FRAC = 0.82
MIN_ITEM_AREA_FRAC = 0.015
MAX_ITEM_AREA_FRAC = 0.80
FOCUS_SMOOTHING = 0.28
FOCUS_MAX_MISSES = 8
FOCUS_DETECTION_INTERVAL = 0.10
FOCUS_DEVICE = os.environ.get("FOCUS_DEVICE", "cpu")
FOCUS_IMGSZ = int(os.environ.get("FOCUS_IMGSZ", "640"))


def _rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgb_to_bgr(rgb):
    r, g, b = rgb
    return (b, g, r)


def compute_roi(w, h):
    """Centered guide-box rectangle (x1, y1, x2, y2) for a w x h frame."""
    bw, bh = int(w * ROI_W_FRAC), int(h * ROI_H_FRAC)
    x1 = (w - bw) // 2
    y1 = (h - bh) // 2
    return x1, y1, x1 + bw, y1 + bh


def draw_guide_box(frame, color_rgb=GUIDE_COLOR):
    """Draw a centered guide box with corner brackets + a small hint label."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = compute_roi(w, h)
    color = _rgb_to_bgr(color_rgb)
    # Corner brackets (cleaner than a full rectangle).
    L = max(20, int(min(w, h) * 0.06))   # bracket arm length
    t = 3                                # thickness
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)
    cv2.putText(frame, "Hold item in the box", (x1, max(y1 - 12, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return x1, y1, x2, y2


def _box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def _smooth_box(previous, current, alpha=FOCUS_SMOOTHING):
    if previous is None:
        return current
    return tuple(
        int(round((1.0 - alpha) * p + alpha * c))
        for p, c in zip(previous, current)
    )


def detect_held_item_box(frame, previous_box=None):
    """Find a likely held item in the center of the frame using local CV only.

    This is deliberately not an LLM/model step. It just gives the live preview a
    helpful visual focus box, and the returned coordinates are never sent to
    Gemini.
    """
    h, w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = compute_roi(w, h)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]
    roi_area = float(roi_w * roi_h)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(edges, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    center_x, center_y = roi_w / 2.0, roi_h / 2.0
    max_dist = (center_x ** 2 + center_y ** 2) ** 0.5
    previous_roi_box = None
    if previous_box is not None:
        px1, py1, px2, py2 = previous_box
        previous_roi_box = (px1 - rx1, py1 - ry1, px2 - rx1, py2 - ry1)

    best = None
    best_score = 0.0
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_frac = (bw * bh) / roi_area
        if area_frac < MIN_ITEM_AREA_FRAC or area_frac > MAX_ITEM_AREA_FRAC:
            continue
        box_cx, box_cy = x + bw / 2.0, y + bh / 2.0
        dist = ((box_cx - center_x) ** 2 + (box_cy - center_y) ** 2) ** 0.5
        centrality = 1.0 - min(dist / max_dist, 1.0)
        edge_penalty = 0.0
        if x <= 2 or y <= 2 or x + bw >= roi_w - 2 or y + bh >= roi_h - 2:
            edge_penalty = 0.25

        continuity = 0.0
        if previous_roi_box is not None:
            candidate = (x, y, x + bw, y + bh)
            continuity = _box_iou(previous_roi_box, candidate)
            prev_cx, prev_cy = _box_center(previous_roi_box)
            motion = ((box_cx - prev_cx) ** 2 + (box_cy - prev_cy) ** 2) ** 0.5
            if continuity < 0.08 and motion > max_dist * 0.38:
                continue

        score = (
            area_frac * 0.30
            + centrality * 0.35
            + continuity * 0.45
            - edge_penalty
        )
        if score > best_score:
            best = (x, y, x + bw, y + bh)
            best_score = score

    if best is None:
        return None

    x1, y1, x2, y2 = best
    pad = max(8, int(min(roi_w, roi_h) * 0.025))
    return (
        max(0, rx1 + x1 - pad),
        max(0, ry1 + y1 - pad),
        min(w, rx1 + x2 + pad),
        min(h, ry1 + y2 + pad),
    )


def draw_item_highlight(frame, box, label=None):
    """Draw the local preview-only item highlight."""
    if box is None:
        draw_guide_box(frame)
        return None

    x1, y1, x2, y2 = box
    color = _rgb_to_bgr(ITEM_HIGHLIGHT_COLOR)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    caption = "Finding item..." if label == "loading" else "Item to categorize"
    cv2.putText(frame, caption, (x1, max(y1 - 12, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return box


# ===========================================================================
# Background camera thread (live preview only -- no model runs here)
# ===========================================================================
class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.error = None
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            self.error = (f"Could not open webcam (index {config.CAMERA_INDEX}). "
                          f"Close other apps using the camera or change "
                          f"CAMERA_INDEX in waste_sorter/config.py.")
            return
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)  # mirror, like a selfie cam
            with self.lock:
                self.latest_frame = frame
        cap.release()

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self._running = False


class LocalFocusThread(threading.Thread):
    """Runs the local detector for preview boxes only.

    Gemini still receives a clean frame in _on_capture; this thread only updates
    the UI's item-focus box.
    """

    def __init__(self, camera):
        super().__init__(daemon=True)
        self.camera = camera
        self.lock = threading.Lock()
        self.latest_box = None
        self.latest_label = None
        self.model_ready = False
        self.error = None
        self._running = True

    def run(self):
        try:
            from waste_sorter.detector import WasteDetector
            detector = WasteDetector(device=FOCUS_DEVICE, imgsz=FOCUS_IMGSZ)
            self.model_ready = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return

        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            try:
                detection = detector.detect(frame)
            except Exception as exc:  # noqa: BLE001
                print(f"[focus] detection error: {exc}")
                detection = None

            with self.lock:
                if detection is None:
                    self.latest_box = None
                    self.latest_label = None
                else:
                    self.latest_box = detection.box
                    self.latest_label = detection.label

            time.sleep(FOCUS_DETECTION_INTERVAL)

    def snapshot(self):
        with self.lock:
            return self.latest_box, self.latest_label, self.model_ready, self.error

    def stop(self):
        self._running = False


# ===========================================================================
# The Tkinter application
# ===========================================================================
class LLMSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trash Sorter — LLM mode")
        self.root.configure(bg=THEME_BG)

        self.camera = CameraThread()
        self.local_focus = LocalFocusThread(self.camera)
        self.classifier = None
        self.client_error = None

        self.mode = "live"            # "live" or "frozen"
        self.captured_frame = None    # full frame shown when frozen
        self.analyzing = False
        self.result = None
        self._result_time = 0.0
        self.photo = None
        self.focus_box = None
        self.focus_misses = 0
        self.captured_focus_box = None
        self.captured_focus_label = None

        self._build_ui()

        self.camera.start()
        self.local_focus.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(30, self._tick)

    # ---------------------------------------------------------------- UI build
    def _make_button(self, parent, text, command, bg, active_bg):
        """A Label styled as a button. tk.Button ignores bg colours on macOS,
        so we use a Label (which respects colours everywhere) for readable,
        themed buttons."""
        btn = tk.Label(parent, text=text, font=("Helvetica", 14, "bold"),
                       bg=bg, fg="white", padx=16, pady=9, cursor="hand2")
        btn.bind("<Button-1>", lambda _e: command())
        btn.bind("<Enter>", lambda _e: btn.config(bg=active_bg))
        btn.bind("<Leave>", lambda _e: btn.config(bg=bg))
        return btn

    def _build_ui(self):
        tk.Label(self.root, text="Trash Sorter — LLM mode",
                 font=("Helvetica", 22, "bold"),
                 bg=THEME_BG, fg=THEME_TEXT).grid(
                     row=0, column=0, columnspan=2, pady=(14, 2))
        tk.Label(self.root,
                 text="Hold up one item, then press Capture.",
                 font=("Helvetica", 12), bg=THEME_BG,
                 fg=THEME_SUBTLE).grid(row=1, column=0, columnspan=2, pady=(0, 10))

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=2, column=0, padx=16, pady=8)

        info = tk.Frame(self.root, bg=THEME_BG)
        info.grid(row=2, column=1, padx=16, pady=8, sticky="n")

        self.status_label = tk.Label(info, text="Live preview",
                                     font=("Helvetica", 17, "bold"),
                                     bg=THEME_BG, fg=THEME_TEXT,
                                     wraplength=320, justify="left")
        self.status_label.pack(anchor="w", pady=(0, 12))

        self.object_label = tk.Label(info, text="Item: —", font=("Helvetica", 13),
                                     bg=THEME_BG, fg=THEME_SUBTLE,
                                     wraplength=320, justify="left")
        self.object_label.pack(anchor="w")
        self.category_label = tk.Label(info, text="Bin: —", font=("Helvetica", 13),
                                       bg=THEME_BG, fg=THEME_SUBTLE)
        self.category_label.pack(anchor="w")
        self.confidence_label = tk.Label(info, text="Confidence: —",
                                         font=("Helvetica", 13), bg=THEME_BG,
                                         fg=THEME_SUBTLE)
        self.confidence_label.pack(anchor="w")
        self.reason_label = tk.Label(info, text="", font=("Helvetica", 11, "italic"),
                                     bg=THEME_BG, fg=THEME_SUBTLE,
                                     wraplength=320, justify="left")
        self.reason_label.pack(anchor="w", pady=(6, 14))

        self.capture_btn = self._make_button(info, "Capture & Classify",
                                             self._on_capture,
                                             BTN_PRIMARY, BTN_PRIMARY_ACTIVE)
        self.capture_btn.pack(anchor="w", pady=(0, 8))
        self.live_btn = self._make_button(info, "Resume Live",
                                          self._on_resume_live,
                                          BTN_SECONDARY, BTN_SECONDARY_ACTIVE)
        self.live_btn.pack(anchor="w")

        bins = tk.Frame(self.root, bg=THEME_BG)
        bins.grid(row=3, column=0, columnspan=2, pady=(8, 18))
        self.bin_cards = {}
        for i, cat in enumerate(config.CATEGORIES):
            disp = config.CATEGORY_DISPLAY[cat]
            card = tk.Frame(bins, bg=THEME_PANEL, width=190, height=110,
                            highlightthickness=3, highlightbackground=THEME_PANEL)
            card.grid(row=0, column=i, padx=10)
            card.grid_propagate(False)
            name = tk.Label(card, text=disp["name"], font=("Helvetica", 15, "bold"),
                            bg=THEME_PANEL, fg=THEME_TEXT)
            name.pack(pady=(18, 2))
            hint = tk.Label(card, text=disp["hint"], font=("Helvetica", 9),
                            wraplength=170, bg=THEME_PANEL, fg=THEME_SUBTLE)
            hint.pack()
            self.bin_cards[cat] = {"frame": card, "name": name, "hint": hint}

    # ----------------------------------------------------------- button actions
    def _on_capture(self):
        if self.analyzing:
            return
        frame = self.camera.get_frame()
        if frame is None:
            return
        self.captured_frame = frame
        self.captured_focus_box = self.focus_box
        self.captured_focus_label = None
        self.mode = "frozen"
        self.result = None
        self.analyzing = True
        # Send a clean camera frame. Preview highlights are drawn later on a
        # display copy only, so Gemini never sees the UI box.
        gemini_frame = frame.copy()
        threading.Thread(target=self._classify_worker, args=(gemini_frame,),
                         daemon=True).start()

    def _on_resume_live(self):
        self.mode = "live"
        self.result = None
        self.analyzing = False
        self.focus_box = None
        self.focus_misses = 0
        self.captured_focus_box = None
        self.captured_focus_label = None

    def _classify_worker(self, frame):
        if self.classifier is None and self.client_error is None:
            try:
                self.classifier = LLMClassifier()
            except Exception as exc:  # noqa: BLE001
                self.client_error = str(exc)
        if self.client_error:
            self.result = {"object": "", "category": None, "category_name": "Unsure",
                           "confidence": 0, "reason": "", "error": self.client_error}
        else:
            self.result = self.classifier.classify(frame)
        self._result_time = time.monotonic()
        self.analyzing = False

    # ------------------------------------------------------------- main loop
    def _tick(self):
        if self.camera.error:
            self.status_label.config(text=self.camera.error, fg="#ff9b9b")
            self.root.after(200, self._tick)
            return

        frame = self.camera.get_frame() if self.mode == "live" else self.captured_frame
        if frame is not None:
            self._render_frame(frame)

        self._update_readouts()
        self._update_bins()
        self.root.after(30, self._tick)

    def _render_frame(self, frame):
        frame = frame.copy()

        # Draw the item highlight on this display copy only. The image sent to
        # Gemini is prepared in _on_capture before any overlay drawing happens.
        if self.mode == "frozen":
            focus_box = self.captured_focus_box
            focus_label = self.captured_focus_label
        else:
            focus_box, focus_label = self._next_focus(frame)
        draw_item_highlight(frame, focus_box, focus_label)

        # On a finished capture, draw a result banner across the bottom.
        if self.mode == "frozen" and self.result and not self.analyzing:
            cat = self.result["category"]
            color = config.CATEGORY_COLOR.get(cat, config.NEUTRAL_COLOR)
            caption = f"{self.result['object']}  ->  {self.result['category_name']}"
            if self.result.get("confidence"):
                caption += f"  ({self.result['confidence']}%)"
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 46), (w, h), _rgb_to_bgr(color), -1)
            cv2.putText(frame, caption, (14, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)

        h, w = frame.shape[:2]
        scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=self.photo)

    def _next_focus(self, frame):
        local_box, local_label, local_ready, local_error = self.local_focus.snapshot()
        if local_ready and local_box is not None:
            candidate = local_box
            label = local_label
        elif local_error:
            candidate = detect_held_item_box(frame, self.focus_box)
            label = None
        else:
            candidate = self.focus_box
            label = "loading"

        if candidate is None:
            self.focus_misses += 1
            if self.focus_misses <= FOCUS_MAX_MISSES:
                return self.focus_box, None
            self.focus_box = None
            return None, None

        self.focus_misses = 0
        self.focus_box = _smooth_box(self.focus_box, candidate)
        return self.focus_box, label

    def _update_readouts(self):
        if self.mode == "live":
            self.status_label.config(text="Live preview", fg=THEME_TEXT)
            self.object_label.config(text="Item: —")
            self.category_label.config(text="Bin: —")
            self.confidence_label.config(text="Confidence: —")
            self.reason_label.config(text="")
            return

        if self.analyzing:
            self.status_label.config(text="Analyzing… asking the model", fg="#ffd166")
            self.object_label.config(text="Item: …")
            self.category_label.config(text="Bin: …")
            self.confidence_label.config(text="Confidence: …")
            self.reason_label.config(text="")
            return

        r = self.result or {}
        if r.get("error"):
            self.status_label.config(text="Couldn't classify", fg="#ff9b9b")
            self.object_label.config(text=f"Error: {r['error']}")
            self.category_label.config(text="Bin: —")
            self.confidence_label.config(text="Confidence: —")
            self.reason_label.config(text="")
            return

        cat = r.get("category")
        name = r.get("category_name", "Unsure")
        if cat is not None:
            self.status_label.config(text=f"➜  Place in {name}!", fg="#9dffb0")
        else:
            self.status_label.config(text="Unsure — try capturing again", fg="#ffd166")
        self.object_label.config(text=f"Item: {r.get('object', '—')}")
        self.category_label.config(text=f"Bin: {name}")
        self.confidence_label.config(text=f"Confidence: {r.get('confidence', 0)}%")
        self.reason_label.config(text=r.get("reason", ""))

    def _update_bins(self):
        target = None
        if self.mode == "frozen" and self.result and not self.analyzing:
            target = self.result.get("category")
        fresh = (time.monotonic() - self._result_time) < 3.0
        blink_on = int(time.time() * 3) % 2 == 0

        for c, w in self.bin_cards.items():
            if c == target:
                base = config.CATEGORY_COLOR[c]
                border = _rgb_to_hex(base) if (not fresh or blink_on) else "#ffffff"
                self._set_card(w, _rgb_to_hex(base), border)
            else:
                self._set_card(w, THEME_PANEL, THEME_PANEL)

    @staticmethod
    def _set_card(w, fill, border):
        w["frame"].config(bg=fill, highlightbackground=border, highlightcolor=border)
        w["name"].config(bg=fill)
        w["hint"].config(bg=fill)

    def _on_close(self):
        self.local_focus.stop()
        self.camera.stop()
        self.root.after(150, self.root.destroy)


def main():
    root = tk.Tk()
    LLMSorterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
