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
  * Purple kiosk theme.
  * A local preview-only highlight box is drawn on the live feed to help you aim.
    The box is only a UI overlay; it is never drawn onto the image sent to the
    LLM.
"""

import threading
import time
import os
import queue
import shutil
import subprocess
from difflib import SequenceMatcher
from collections import deque

import cv2
from PIL import Image, ImageTk
import tkinter as tk

from waste_sorter import config
from .llm_client import LLMClassifier


# ---- "Ask Oscar" cream-paper theme ----
THEME_BG = "#fdf6e3"              # warm cream paper background
THEME_PANEL = "#fff4cc"           # speech-bubble cream
THEME_TEXT = "#1d1d1d"
THEME_SUBTLE = "#4f4f4f"
TITLE_GREEN = "#3aa84b"           # green "ASK OSCAR." headline
SPEECH_BG = "#fff3b0"             # speech bubble fill
SPEECH_BORDER = "#e09a2a"         # warm orange bubble outline
SPEECH_TEXT = "#2a1a06"
HINT_BG = "#f0c860"               # slightly darker amber sub-hint
HINT_BORDER = "#b07720"
HINT_TEXT = "#2a1a06"
BTN_PRIMARY = "#7b5bd6"
BTN_PRIMARY_ACTIVE = "#9274ee"
BTN_SECONDARY = "#7a6a5a"
BTN_SECONDARY_ACTIVE = "#9a8a78"
VOICE_CALLOUT_BG = "#f2b84b"
VOICE_CALLOUT_FG = "#241331"
GUIDE_COLOR = (210, 180, 255)
ITEM_HIGHLIGHT_COLOR = (104, 255, 188)  # mint green (RGB), local UI only
ATTENTION_RED = "#d90429"
ATTENTION_RED_RGB = (217, 4, 41)

# Override config colors for the chunky rounded "bin pills" along the bottom,
# so they match the sketch: green / teal / red-orange.
BIN_PILL_COLOR = {
    "COMPOST": "#3aa84b",
    "RECYCLING": "#2da3a8",
    "LANDFILL": "#e04e2a",
}

# Path to the Oscar mascot image (transparent PNG sitting next to this file).
OSCAR_PNG_PATH = os.path.join(os.path.dirname(__file__), "assets", "oscar.png")

# Guide box size as a fraction of the frame (centered). This is only used as a
# local search area for the preview highlight.
ROI_W_FRAC = 0.66
ROI_H_FRAC = 0.82
MIN_ITEM_AREA_FRAC = 0.015
MAX_ITEM_AREA_FRAC = 0.80
LLM_USE_FOCUS_CROP = os.environ.get("LLM_USE_FOCUS_CROP", "1").lower() not in {
    "0", "false", "off", "no",
}
LLM_CROP_PADDING_FRAC = float(os.environ.get("LLM_CROP_PADDING_FRAC", "0.18"))
FOCUS_SMOOTHING = 0.28
FOCUS_MAX_MISSES = 8
FOCUS_DETECTION_INTERVAL = 0.10
FOCUS_DEVICE = os.environ.get("FOCUS_DEVICE", "cpu")
FOCUS_IMGSZ = int(os.environ.get("FOCUS_IMGSZ", "640"))
VOICE_TRIGGER_ENABLED = os.environ.get("VOICE_TRIGGER", "1").lower() not in {
    "0", "false", "off", "no",
}
VOICE_TRIGGER_PHRASES = [
    phrase.strip().lower()
    for phrase in os.environ.get(
        "VOICE_TRIGGER_PHRASES",
        os.environ.get(
            "VOICE_TRIGGER_PHRASE",
            # "hey oscar" is the primary phrase shown in the UI. The rest are
            # accepted variants the Google recognizer commonly returns instead.
            "hey oscar,oscar help,oscar scan,oscar please,"
            "hey oscar help,hey oscar scan,ask oscar,okay oscar",
        ),
    ).split(",")
    if phrase.strip()
]
# Loose match fallback: any heard phrase that pairs an "oscar"-style subject
# word with one of these intent words also fires the trigger. This catches
# recognizer slips like "asked her help" or "oscars scan".
VOICE_INTENT_WORDS = {"help", "sort", "scan", "classify", "please",
                      "go", "look", "check", "hey", "ask"}
VOICE_SUBJECT_WORDS = {"oscar", "oscars", "oskar", "ossker", "ascar",
                       "oscarr", "ostar"}
VOICE_INPUT_DEVICE = os.environ.get("VOICE_INPUT_DEVICE", "").strip()
VOICE_INPUT_DEVICE_KEYWORDS = [
    word.strip().lower()
    for word in os.environ.get(
        "VOICE_INPUT_DEVICE_KEYWORDS",
        "webcam,camera,cam,nexigo,usb",
    ).split(",")
    if word.strip()
]
VOICE_SAMPLE_RATE = int(os.environ.get("VOICE_SAMPLE_RATE", "16000"))
VOICE_WINDOW_SECONDS = float(os.environ.get("VOICE_WINDOW_SECONDS", "2.2"))
VOICE_CHECK_INTERVAL = float(os.environ.get("VOICE_CHECK_INTERVAL", "0.45"))
VOICE_COOLDOWN_SECONDS = float(os.environ.get("VOICE_COOLDOWN_SECONDS", "3.0"))
RESULT_HOLD_SECONDS = float(os.environ.get("RESULT_HOLD_SECONDS", "10.0"))
TTS_ENABLED = os.environ.get("TEXT_TO_SPEECH", "1").lower() not in {
    "0", "false", "off", "no",
}
TTS_VOICE = os.environ.get("TTS_VOICE", "").strip()
TTS_RATE = int(os.environ.get("TTS_RATE", "180"))
AUTO_SCAN_ENABLED = os.environ.get("AUTO_SCAN", "1").lower() not in {
    "0", "false", "off", "no",
}
AUTO_SCAN_SECONDS = float(os.environ.get("AUTO_SCAN_SECONDS", "5.0"))
AUTO_SCAN_MOTION_TOLERANCE = float(os.environ.get("AUTO_SCAN_MOTION_TOLERANCE", "0.08"))
FULLSCREEN_DEFAULT = os.environ.get("FULLSCREEN", "0").lower() in {
    "1", "true", "on", "yes",
}


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


def _box_motion_ratio(a, b):
    if a is None or b is None:
        return 1.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = _box_center(a)
    bcx, bcy = _box_center(b)
    diag = max(((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5, 1.0)
    center_move = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / diag
    size_change = (
        abs((ax2 - ax1) - (bx2 - bx1)) + abs((ay2 - ay1) - (by2 - by1))
    ) / max((ax2 - ax1) + (ay2 - ay1), 1.0)
    return center_move + size_change


def _smooth_box(previous, current, alpha=FOCUS_SMOOTHING):
    if previous is None:
        return current
    return tuple(
        int(round((1.0 - alpha) * p + alpha * c))
        for p, c in zip(previous, current)
    )


def crop_frame_to_box(frame, box, padding_frac=LLM_CROP_PADDING_FRAC):
    """Return a clean crop around the detector box, or the full frame if invalid."""
    if box is None or frame is None:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 8 or bh <= 8:
        return frame

    pad = int(round(max(bw, bh) * padding_frac))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2].copy()


def detect_held_item_box(frame, previous_box=None):
    """Find a likely held item in the center of the frame using local CV only.

    This just gives the live preview a helpful visual focus box. On capture,
    the app can use this box to crop the clean image sent to Gemini.
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
        print(f"[camera] Opening camera index {config.CAMERA_INDEX}")
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

    This thread updates the UI item-focus box. On capture, that same box may be
    used to crop the clean image sent to Gemini.
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


class VoiceTriggerThread(threading.Thread):
    """Continuously listens for a spoken phrase and asks the UI to capture.

    This uses the optional `speech_recognition` + `sounddevice` packages. The
    speech-to-text step uses SpeechRecognition's Google Web Speech recognizer,
    so rolling audio snippets leave the machine when this trigger is enabled.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.status = "Voice trigger off"
        self.error = None
        self.last_heard = ""
        self._running = VOICE_TRIGGER_ENABLED
        self._triggered = False
        self._audio_queue = queue.Queue()

    @staticmethod
    def _normalize(text):
        return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace())

    @staticmethod
    def _word_close(a, b, threshold=0.78):
        if a == b:
            return True
        if len(a) < 3 or len(b) < 3:
            return False
        return SequenceMatcher(None, a, b).ratio() >= threshold

    @classmethod
    def _has_close_word(cls, words, choices):
        return any(cls._word_close(word, choice) for word in words for choice in choices)

    @classmethod
    def _matches_trigger(cls, heard, targets):
        compact_heard = heard.replace(" ", "")
        for target in targets:
            if target in heard or target.replace(" ", "") in compact_heard:
                return True

        words = heard.split()
        if not words:
            return False

        # Allows recognition misses like "crash can", "trash cam",
        # "trash scan", or "garbage help" without accepting arbitrary speech.
        return (
            cls._has_close_word(words, VOICE_SUBJECT_WORDS)
            and cls._has_close_word(words, VOICE_INTENT_WORDS)
        )

    @staticmethod
    def _input_devices(sd):
        return [
            (i, dev)
            for i, dev in enumerate(sd.query_devices())
            if dev.get("max_input_channels", 0) > 0
        ]

    @classmethod
    def _resolve_input_device(cls, sd):
        devices = cls._input_devices(sd)
        if not devices:
            return None, "system default"

        if VOICE_INPUT_DEVICE:
            try:
                wanted_index = int(VOICE_INPUT_DEVICE)
            except ValueError:
                wanted_index = None

            for index, dev in devices:
                name = str(dev.get("name", ""))
                if wanted_index == index or VOICE_INPUT_DEVICE.lower() in name.lower():
                    return index, name

        for keyword in VOICE_INPUT_DEVICE_KEYWORDS:
            for index, dev in devices:
                name = str(dev.get("name", ""))
                if keyword in name.lower():
                    return index, name

        try:
            default_index = sd.default.device[0]
        except Exception:  # noqa: BLE001
            default_index = None
        for index, dev in devices:
            if index == default_index:
                return index, str(dev.get("name", "system default"))

        return None, "system default"

    def run(self):
        if not VOICE_TRIGGER_ENABLED:
            return

        try:
            import sounddevice as sd
            import speech_recognition as sr
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.status = "Voice trigger unavailable"
                self.error = (
                    "Install voice dependencies with: "
                    "pip install -r requirements-llm.txt")
                self.last_heard = str(exc)
            return

        recognizer = sr.Recognizer()
        recognizer.operation_timeout = 1.8
        targets = [self._normalize(phrase) for phrase in VOICE_TRIGGER_PHRASES]
        cooldown_until = 0.0
        max_audio_bytes = int(VOICE_SAMPLE_RATE * VOICE_WINDOW_SECONDS) * 2
        audio_buffer = deque(maxlen=max_audio_bytes)
        next_check = 0.0
        input_device, input_device_name = self._resolve_input_device(sd)
        listening_status = f'Voice: listening on {input_device_name}'

        with self.lock:
            self.status = listening_status
        print(f"[voice] Using input device: {input_device_name}")

        def audio_callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                with self.lock:
                    self.status = f"Voice mic warning: {status}"
            self._audio_queue.put(bytes(indata))

        try:
            with sd.RawInputStream(
                device=input_device,
                samplerate=VOICE_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=int(VOICE_SAMPLE_RATE * 0.10),
                callback=audio_callback,
            ):
                while self._running:
                    try:
                        chunk = self._audio_queue.get(timeout=0.12)
                    except queue.Empty:
                        chunk = None

                    if chunk:
                        audio_buffer.extend(chunk)

                    now = time.monotonic()
                    if (
                        len(audio_buffer) < int(VOICE_SAMPLE_RATE * 0.8) * 2
                        or now < next_check
                        or now < cooldown_until
                    ):
                        continue

                    next_check = now + VOICE_CHECK_INTERVAL
                    try:
                        audio_bytes = bytes(audio_buffer)
                        audio_data = sr.AudioData(audio_bytes, VOICE_SAMPLE_RATE, 2)
                        text = recognizer.recognize_google(audio_data)
                        heard = self._normalize(text)
                        with self.lock:
                            self.last_heard = text

                        if self._matches_trigger(heard, targets):
                            with self.lock:
                                self._triggered = True
                                self.status = "Voice trigger heard"
                            cooldown_until = now + VOICE_COOLDOWN_SECONDS
                            audio_buffer.clear()
                        else:
                            with self.lock:
                                self.status = listening_status
                    except sr.UnknownValueError:
                        with self.lock:
                            self.status = listening_status
                    except Exception as exc:  # noqa: BLE001
                        with self.lock:
                            self.status = "Voice trigger paused"
                            self.error = str(exc)
                        time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.status = "Voice trigger unavailable"
                self.error = str(exc)

    def consume_trigger(self):
        with self.lock:
            if not self._triggered:
                return False
            self._triggered = False
            return True

    def snapshot(self):
        with self.lock:
            return self.status, self.error, self.last_heard

    def stop(self):
        self._running = False


# ===========================================================================
# Speech bubble — a Canvas-based comic bubble that scales with its parent.
# ===========================================================================
class SpeechBubble:
    """A rounded comic-style speech bubble that holds wrapped, centred text
    and grows to fit its width.

    Drawn entirely on a ``tk.Canvas`` so the bubble shape, the tail pointing
    down to Oscar, and the text all rescale together every time the container
    changes size. Use ``set(...)`` to change the text or colours; the bubble
    redraws itself.
    """

    # Default geometry
    _CORNER_RADIUS = 26
    _STROKE = 5
    _PAD_X = 28
    _PAD_Y = 22
    _TAIL_W = 64       # base width
    _TAIL_H = 38       # how far the tail dips below the body
    _TAIL_LEAN = 0.18  # how far the tail's tip sits from centre, as a
                       # fraction of TAIL_W (positive = points toward Oscar
                       # which is below+left)

    def __init__(self, parent, *, text="", fill, border, fg,
                 font=("Helvetica", 22, "bold"),
                 page_bg, min_height=140):
        self._text = text
        self._fill = fill
        self._border = border
        self._fg = fg
        self._font = font
        self._page_bg = page_bg
        self._min_height = min_height

        # The canvas itself. We set bg = parent's bg so the area outside the
        # bubble fades into the page.
        self.canvas = tk.Canvas(
            parent, bg=page_bg, highlightthickness=0,
            height=min_height,
        )
        self.canvas.bind("<Configure>", self._on_configure)
        # When we receive a resize event the redraw also resizes the canvas
        # height to fit text; debounce so we don't infinite-loop.
        self._redraw_scheduled = False
        self._last_drawn_size = (0, 0)

    # -- public API -------------------------------------------------------
    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)

    def pack_forget(self):
        self.canvas.pack_forget()

    def configure(self, *, page_bg=None):
        if page_bg is not None and page_bg != self._page_bg:
            self._page_bg = page_bg
            self.canvas.configure(bg=page_bg)
            self._schedule_redraw()

    def set(self, *, text=None, fill=None, border=None, fg=None):
        changed = False
        if text is not None and text != self._text:
            self._text = text
            changed = True
        if fill is not None and fill != self._fill:
            self._fill = fill
            changed = True
        if border is not None and border != self._border:
            self._border = border
            changed = True
        if fg is not None and fg != self._fg:
            self._fg = fg
            changed = True
        if changed:
            self._schedule_redraw()

    # -- internals --------------------------------------------------------
    def _on_configure(self, _event):
        self._schedule_redraw()

    def _schedule_redraw(self):
        if self._redraw_scheduled:
            return
        self._redraw_scheduled = True
        # Defer to idle so multiple Configure events collapse into one draw.
        self.canvas.after_idle(self._redraw)

    def _redraw(self):
        self._redraw_scheduled = False
        c = self.canvas
        w = c.winfo_width()
        if w < 60:
            # Canvas hasn't been measured yet -- try again shortly.
            c.after(50, self._schedule_redraw)
            return

        pad_x, pad_y = self._PAD_X, self._PAD_Y
        stroke = self._STROKE
        radius = self._CORNER_RADIUS
        tail_w, tail_h = self._TAIL_W, self._TAIL_H

        body_left = stroke
        body_right = w - stroke
        body_top = stroke
        text_max_w = max(40, (body_right - body_left) - 2 * pad_x)

        # Measure wrapped text height by drawing a hidden item, then deleting.
        probe = c.create_text(
            0, 0, text=self._text or " ", font=self._font,
            width=text_max_w, anchor="nw", justify="center",
        )
        bbox = c.bbox(probe)
        c.delete(probe)
        text_h = (bbox[3] - bbox[1]) if bbox else 30

        body_bottom = body_top + text_h + 2 * pad_y
        total_h = int(body_bottom + tail_h + stroke + 2)
        total_h = max(self._min_height, total_h)

        # Resize the canvas to match. Skip if it's already the right size to
        # avoid bouncing Configure events.
        if (w, total_h) != self._last_drawn_size:
            c.configure(height=total_h)
        self._last_drawn_size = (w, total_h)

        # Now do the actual drawing.
        c.delete("all")
        self._draw_body(c, body_left, body_top, body_right, body_bottom,
                        radius)
        self._draw_tail(c, body_left, body_right, body_bottom, tail_w, tail_h)
        # Cover the seam where tail meets body so the body's bottom border
        # doesn't run through the tail's open base.
        tail_cx = (body_left + body_right) / 2
        seam_x1 = tail_cx - tail_w / 2 + stroke * 0.6
        seam_x2 = tail_cx + tail_w / 2 - stroke * 0.6
        c.create_line(seam_x1, body_bottom, seam_x2, body_bottom,
                      fill=self._fill, width=stroke + 2)
        # Finally the wrapped, centred text.
        cx = (body_left + body_right) / 2
        cy = (body_top + body_bottom) / 2
        c.create_text(cx, cy, text=self._text, font=self._font,
                      fill=self._fg, width=text_max_w,
                      justify="center", anchor="center")

    @staticmethod
    def _rounded_rect_points(x1, y1, x2, y2, r):
        """Return a point list for a smooth rounded-rectangle polygon."""
        # The 'smooth' polygon interpolates between consecutive points. By
        # placing two points right at the corner (the line-end + the curve
        # control point), the corner stays tight against the rectangle edges.
        return [
            x1 + r, y1,  x2 - r, y1,
            x2, y1,      x2, y1 + r,
            x2, y2 - r,
            x2, y2,      x2 - r, y2,
            x1 + r, y2,
            x1, y2,      x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]

    def _draw_body(self, c, x1, y1, x2, y2, r):
        pts = self._rounded_rect_points(x1, y1, x2, y2, r)
        c.create_polygon(pts, smooth=True, splinesteps=24,
                         fill=self._fill, outline=self._border,
                         width=self._STROKE)

    def _draw_tail(self, c, body_left, body_right, body_bottom,
                   tail_w, tail_h):
        cx = (body_left + body_right) / 2
        lean = tail_w * self._TAIL_LEAN
        # Triangle: two base points on the body's bottom edge, one tip below.
        pts = [
            cx - tail_w / 2, body_bottom,
            cx + tail_w / 2, body_bottom,
            cx - lean,       body_bottom + tail_h,
        ]
        c.create_polygon(pts, fill=self._fill, outline=self._border,
                         width=self._STROKE, joinstyle="round")


class _SpeechBubbleShim:
    """Tiny compatibility shim so any leftover ``status_label.config(...)``
    calls (text/fg/bg keywords) translate into bubble updates instead of
    blowing up. New code should call ``app._set_speech(...)`` directly."""

    def __init__(self, bubble):
        self._bubble = bubble

    def config(self, **kwargs):
        text = kwargs.get("text")
        fg = kwargs.get("fg") or kwargs.get("foreground")
        fill = kwargs.get("bg") or kwargs.get("background")
        self._bubble.set(text=text, fg=fg, fill=fill)

    configure = config


# ===========================================================================
# The Tkinter application
# ===========================================================================
class LLMSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Campus Waste Sorting Assistant — LLM Mode")
        self.root.configure(bg=THEME_BG)
        self.fullscreen = FULLSCREEN_DEFAULT
        self.root.attributes("-fullscreen", self.fullscreen)
        self.root.minsize(1000, 760)
        self.root.bind("<F11>", lambda _e: self._toggle_fullscreen())
        self.root.bind("<Escape>", lambda _e: self._set_fullscreen(False))

        self.camera = CameraThread()
        self.local_focus = LocalFocusThread(self.camera)
        self.voice_trigger = VoiceTriggerThread()
        self.classifier = None
        self.client_error = None

        self.mode = "live"            # "live" or "frozen"
        self.captured_frame = None    # full frame shown when frozen
        self.analyzing = False
        self.result = None
        self._result_time = 0.0
        self._auto_resume_done = False
        self.photo = None
        self.focus_box = None
        self.focus_misses = 0
        self.captured_focus_box = None
        self.captured_focus_label = None
        self.last_focus_box = None
        self.still_reference_box = None
        self.still_started_at = None
        self.auto_scan_remaining = None
        self.speaking = False
        self.tts_lock = threading.Lock()
        self.bg_widgets = []          # widgets that follow the page background
        self._oscar_photo = None      # PhotoImage reference (don't let GC kill it)
        self._oscar_load_error = None

        self._build_ui()

        self.camera.start()
        self.local_focus.start()
        self.voice_trigger.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(30, self._tick)

    # ---------------------------------------------------------------- UI build
    def _make_button(self, parent, text, command, bg, active_bg):
        """A Label styled as a button. tk.Button ignores bg colours on macOS,
        so we use a Label (which respects colours everywhere) for readable,
        themed buttons."""
        btn = tk.Label(parent, text=text, font=("Helvetica", 20, "bold"),
                       bg=bg, fg="white", padx=22, pady=16, cursor="hand2")
        btn.bind("<Button-1>", lambda _e: command())
        btn.bind("<Enter>", lambda _e: btn.config(bg=active_bg))
        btn.bind("<Leave>", lambda _e: btn.config(bg=bg))
        return btn

    def _build_ui(self):
        # Layout:
        #   row 0: "ASK OSCAR." headline (spans both columns)
        #   row 1: [ left: speech bubble + Oscar PNG ] [ right: webcam ]
        #   row 2: three rounded bin pills spanning both columns
        self.root.grid_columnconfigure(0, weight=2, uniform="cols")  # Oscar
        self.root.grid_columnconfigure(1, weight=3, uniform="cols")  # webcam
        self.root.grid_rowconfigure(1, weight=1)

        # ---- Headline ----
        title = tk.Label(self.root, text="ASK OSCAR!",
                         font=("Helvetica", 56, "bold"),
                         bg=THEME_BG, fg=TITLE_GREEN, anchor="w")
        title.grid(row=0, column=0, columnspan=2, sticky="ew",
                   padx=36, pady=(18, 6))
        self.bg_widgets.append(title)

        # ---- Left column: Oscar with a speech bubble ----
        left = tk.Frame(self.root, bg=THEME_BG)
        left.grid(row=1, column=0, sticky="nsew", padx=(36, 12), pady=(4, 0))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        self.bg_widgets.append(left)

        self._tips_text = (
            'Hold up one item '
            'and wait,\n'
            'or say "hey oscar".'
        )
        # Sub-hint shown beneath the main bubble while we're live. Reminds the
        # user the voice trigger exists when auto-scan is being finicky.
        self._voice_hint_text = (
            'Auto-scan stuck?\n'
            'Say "hey oscar" and I\'ll scan for you.'
        )

        # ----- Left column stack (top -> bottom) ---------------------------
        #   1) Speech bubble (Oscar's words appear here)
        #   2) Oscar PNG (so it visually reads as him speaking)
        #   3) "Auto-scan stuck?" voice hint
        #   4) Detail readouts (Item / Bin / Confidence / Reason)
        # Oscar owns the only flexible row, so the bubble, hint, and readouts
        # keep their reserved space while the mascot resizes between them.
        # rescaled to fit (_resize_oscar_to_fit).

        # 1. Speech bubble.
        self.speech_bubble = SpeechBubble(
            left, text=self._tips_text,
            fill=SPEECH_BG, border=SPEECH_BORDER, fg=SPEECH_TEXT,
            font=("Helvetica", 22, "bold"),
            page_bg=THEME_BG,
        )
        self.speech_bubble.canvas.grid(row=0, column=0, sticky="ew",
                                       pady=(0, 4))
        # Some legacy code paths poke status_label directly (e.g. on a fatal
        # camera error). Route those through _set_speech via this shim.
        self.status_label = _SpeechBubbleShim(self.speech_bubble)

        # 2. Oscar PNG. The grid row grows with the column; the image inside is
        # resampled to match the slot's
        # actual size by _resize_oscar_to_fit (bound to <Configure> on the
        # label itself).
        self._load_oscar_image(max_height=720)
        self.oscar_label = tk.Label(
            left, image=self._oscar_photo, bg=THEME_BG,
            text="" if self._oscar_photo else "(oscar.png missing)",
            font=("Helvetica", 12), fg=THEME_SUBTLE,
            anchor="center",
        )
        self.oscar_label.grid(row=1, column=0, sticky="nsew")
        self.bg_widgets.append(self.oscar_label)
        # Resize when the label's slot changes (fires on column resize, and
        # transitively on root resize, since the slot reflects both).
        self.oscar_label.bind(
            "<Configure>", lambda _e: self._schedule_oscar_resize())
        # First paint hasn't measured yet; trigger one after layout settles.
        self.root.after(200, self._schedule_oscar_resize)

        # 3. "Stuck?" voice hint box -- darker amber, hidden during results.
        self.voice_hint_frame = tk.Frame(
            left, bg=HINT_BG,
            highlightthickness=3,
            highlightbackground=HINT_BORDER,
            highlightcolor=HINT_BORDER,
        )
        self.voice_hint_label = tk.Label(
            self.voice_hint_frame, text=self._voice_hint_text,
            font=("Helvetica", 14, "bold"),
            bg=HINT_BG, fg=HINT_TEXT,
            wraplength=400, justify="center",
            padx=14, pady=10,
        )
        self.voice_hint_label.pack(fill="x")
        self.voice_hint_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self._voice_hint_visible = True

        # 4. Detail readouts -- quiet during live, populated when a result
        # arrives. Sit at the bottom of the column.
        details = tk.Frame(left, bg=THEME_BG)
        details.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        details.grid_columnconfigure(0, weight=1)
        self.details_frame = details
        self.bg_widgets.append(details)

        self.object_label = tk.Label(
            details, text="", font=("Helvetica", 14),
            bg=THEME_BG, fg=THEME_SUBTLE,
            wraplength=420, justify="left", anchor="w",
        )
        self.object_label.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.bg_widgets.append(self.object_label)

        self.category_label = tk.Label(
            details, text="", font=("Helvetica", 16, "bold"),
            bg=THEME_BG, fg=THEME_SUBTLE, anchor="w",
        )
        self.category_label.grid(row=1, column=0, sticky="ew")
        self.bg_widgets.append(self.category_label)

        self.confidence_label = tk.Label(
            details, text="", font=("Helvetica", 14),
            bg=THEME_BG, fg=THEME_SUBTLE, anchor="w",
        )
        self.confidence_label.grid(row=2, column=0, sticky="ew")
        self.bg_widgets.append(self.confidence_label)

        self.reason_label = tk.Label(
            details, text="", font=("Helvetica", 13, "italic"),
            bg=THEME_BG, fg=THEME_SUBTLE,
            wraplength=420, justify="left", anchor="w",
        )
        self.reason_label.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        self.bg_widgets.append(self.reason_label)

        # Kept (hidden) so legacy code paths referencing voice_label don't crash.
        self.voice_label = None

        # ---- Right column: webcam feed + buttons ----
        right = tk.Frame(self.root, bg=THEME_BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(12, 36), pady=4)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        self.bg_widgets.append(right)

        self.video_label = tk.Label(right, bg=THEME_BG)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        btn_row = tk.Frame(right, bg=THEME_BG)
        btn_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.bg_widgets.append(btn_row)
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        # self.capture_btn = self._make_button(
        #     btn_row, "Capture & Classify", self._on_capture,
        #     BTN_PRIMARY, BTN_PRIMARY_ACTIVE,
        # )
        # self.capture_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        # self.live_btn = self._make_button(
        #     btn_row, "Resume Live", self._on_resume_live,
        #     BTN_SECONDARY, BTN_SECONDARY_ACTIVE,
        # )
        # self.live_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # ---- Bottom row: three flashing bin pills ----
        # Tiny top gap so Oscar's trash can touches the recycling pill border.
        bins = tk.Frame(self.root, bg=THEME_BG, height=128)
        bins.grid(row=2, column=0, columnspan=2, padx=36, pady=(0, 4),
                  sticky="ew")
        bins.grid_propagate(False)
        bins.grid_rowconfigure(0, weight=1)
        self.bg_widgets.append(bins)

        self.bin_cards = {}
        for i, cat in enumerate(config.CATEGORIES):
            bins.grid_columnconfigure(i, weight=1, uniform="bins")
            disp = config.CATEGORY_DISPLAY[cat]
            base_color = BIN_PILL_COLOR.get(
                cat, _rgb_to_hex(config.CATEGORY_COLOR.get(cat, (90, 90, 90))))
            canvas = tk.Canvas(
                bins, height=104, bg=THEME_BG, highlightthickness=0,
            )
            canvas.grid(row=0, column=i, padx=10, sticky="nsew")
            self.bin_cards[cat] = {
                "canvas": canvas,
                "name": disp["name"].upper(),
                "color": base_color,
                "fill": base_color,
                "text_color": "#ffffff",
            }
            # Redraw whenever the canvas is resized.
            canvas.bind("<Configure>", lambda _e, c=cat: self._draw_bin_pill(c))

        self.voice_status_label = None

    # ---------------------------------------------------------- Oscar / pills
    def _load_oscar_image(self, max_height=720):
        """Load the Oscar PNG and prepare an initial PhotoImage. The original
        PIL image is kept on ``self._oscar_pil`` so ``_resize_oscar_to_fit``
        can rescale it as the window changes size."""
        try:
            img = Image.open(OSCAR_PNG_PATH).convert("RGBA")
        except Exception as exc:  # noqa: BLE001
            self._oscar_load_error = str(exc)
            self._oscar_photo = None
            self._oscar_pil = None
            print(f"[ui] could not load Oscar PNG: {exc}")
            return
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        self._oscar_pil = img
        w, h = img.size
        scale = min(1.0, max_height / float(h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        self._oscar_photo = ImageTk.PhotoImage(img)
        self._oscar_resize_scheduled = False

    def _schedule_oscar_resize(self):
        """Coalesce repeat Configure events into a single resize call."""
        if getattr(self, "_oscar_resize_scheduled", False):
            return
        self._oscar_resize_scheduled = True
        self.root.after_idle(self._resize_oscar_to_fit)

    def _resize_oscar_to_fit(self):
        """Rescale Oscar to fill the Label's own slot. Since the Oscar Label
        is packed with ``expand=True, fill='both'``, its winfo_width/height
        give the actual space the pack manager has allocated -- much more
        reliable than measuring siblings."""
        self._oscar_resize_scheduled = False
        if not getattr(self, "_oscar_pil", None):
            return
        label = getattr(self, "oscar_label", None)
        if label is None or not label.winfo_exists():
            return

        label.update_idletasks()
        slot_w = label.winfo_width()
        slot_h = label.winfo_height()
        if slot_w <= 1 or slot_h <= 1:
            # Slot hasn't been measured yet -- try again on next idle.
            self.root.after(80, self._schedule_oscar_resize)
            return

        avail_w = max(80, slot_w - 6)
        avail_h = max(80, slot_h - 6)

        src_w, src_h = self._oscar_pil.size
        scale = min(avail_w / src_w, avail_h / src_h)
        # Generous upscale cap so Oscar grows on tall/wide windows without
        # going completely soft.
        scale = min(scale, 3.0)
        new_w = max(40, int(round(src_w * scale)))
        new_h = max(40, int(round(src_h * scale)))

        cur_w = self._oscar_photo.width() if self._oscar_photo else 0
        cur_h = self._oscar_photo.height() if self._oscar_photo else 0
        if abs(new_w - cur_w) < 4 and abs(new_h - cur_h) < 4:
            return

        resized = self._oscar_pil.resize((new_w, new_h), Image.LANCZOS)
        self._oscar_photo = ImageTk.PhotoImage(resized)
        label.config(image=self._oscar_photo)
    def _draw_bin_pill(self, cat):
        """Render a single bin as a rounded pill on its canvas."""
        info = self.bin_cards[cat]
        canvas = info["canvas"]
        fill = info["fill"]
        text_color = info["text_color"]
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 20 or h < 20:
            return
        r = h / 2
        # rounded pill = two circles + a rectangle between them
        canvas.create_oval(0, 0, h, h, fill=fill, outline=fill)
        canvas.create_oval(w - h, 0, w, h, fill=fill, outline=fill)
        canvas.create_rectangle(r, 0, w - r, h, fill=fill, outline=fill)
        canvas.create_text(w / 2, h / 2, text=info["name"],
                           fill=text_color,
                           font=("Helvetica", 26, "bold"))

    def _set_bin_pill(self, cat, fill, text_color="#ffffff"):
        info = self.bin_cards[cat]
        if info["fill"] != fill or info["text_color"] != text_color:
            info["fill"] = fill
            info["text_color"] = text_color
            self._draw_bin_pill(cat)

    def _set_speech(self, text, *, fill=SPEECH_BG, fg=SPEECH_TEXT,
                    border=SPEECH_BORDER):
        """Update the speech-bubble headline + colours in one shot."""
        self.speech_bubble.set(text=text, fill=fill, fg=fg, border=border)

    def _set_voice_hint_visible(self, visible):
        """Show or hide the voice-trigger sub-hint under Oscar."""
        if visible == self._voice_hint_visible:
            return
        if visible:
            self.voice_hint_frame.grid(row=2, column=0, sticky="ew",
                                       pady=(0, 6))
        else:
            self.voice_hint_frame.grid_remove()
        self._voice_hint_visible = visible

    def _set_details_visible(self, visible):
        if visible:
            self.details_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        else:
            self.details_frame.grid_remove()

    # ----------------------------------------------------------- button actions
    def _on_capture(self):
        if self._capture_busy():
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
        self._auto_resume_done = False
        self._reset_auto_scan()
        # Send a clean image focused on YOLO's selected item. Preview overlays
        # are drawn later on a display copy, so Gemini never sees UI graphics.
        gemini_frame = frame.copy()
        if LLM_USE_FOCUS_CROP:
            gemini_frame = crop_frame_to_box(gemini_frame, self.captured_focus_box)
        threading.Thread(target=self._classify_worker, args=(gemini_frame,),
                         daemon=True).start()

    def _on_resume_live(self):
        self.mode = "live"
        self.result = None
        self.analyzing = False
        self._auto_resume_done = False
        self.focus_box = None
        self.focus_misses = 0
        self.captured_focus_box = None
        self.captured_focus_label = None
        self._reset_auto_scan()

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
        self._auto_resume_done = False
        self._speak_result_async(self.result)
        self.analyzing = False

    def _capture_busy(self):
        return self.analyzing or self.speaking or self.mode != "live"

    def _speak_result_async(self, result):
        if not TTS_ENABLED or not result or result.get("error"):
            return
        text = self._speech_text_for_result(result)
        if not text:
            return
        self.speaking = True
        threading.Thread(target=self._speak_text_and_clear, args=(text,),
                         daemon=True).start()

    @staticmethod
    def _speech_text_for_result(result):
        category_name = str(result.get("category_name", "Unsure")).strip()
        reason = " ".join(str(result.get("reason", "")).split())
        if result.get("special"):
            sentence = "Do not throw this in recycling, compost, or landfill"
            if reason:
                sentence += f", because {reason}"
            return sentence
        if not category_name or category_name.lower() == "unsure":
            return "I am not sure where this goes. Please try scanning it again."
        sentence = f"Throw in {category_name.lower()}"
        if reason:
            sentence += f", because {reason}"
        return sentence

    def _speak_text_and_clear(self, text):
        try:
            self._speak_text(text)
        finally:
            self.speaking = False

    def _speak_text(self, text):
        try:
            with self.tts_lock:
                if shutil.which("say"):
                    cmd = ["say", "-r", str(TTS_RATE)]
                    if TTS_VOICE:
                        cmd.extend(["-v", TTS_VOICE])
                    cmd.append(text)
                    subprocess.run(cmd, check=False, timeout=25)
                    return

                try:
                    import pyttsx3
                except ImportError:
                    print("[tts] No text-to-speech engine found.")
                    return

                engine = pyttsx3.init()
                engine.setProperty("rate", TTS_RATE)
                engine.say(text)
                engine.runAndWait()
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] speech error: {exc}")

    # ------------------------------------------------------------- main loop
    def _tick(self):
        if self.camera.error:
            self._set_speech(self.camera.error,
                             fill="#ffd1d1", fg="#7a0000",
                             border="#cc4444")
            self.root.after(200, self._tick)
            return

        if self.voice_trigger.consume_trigger() and not self._capture_busy():
            self._on_capture()

        if (
            self.mode == "live"
            and not self._capture_busy()
            and self._auto_scan_ready()
        ):
            self._on_capture()

        if (
            self.mode == "frozen"
            and self.result
            and not self.analyzing
            and not self._auto_resume_done
            and time.monotonic() - self._result_time > RESULT_HOLD_SECONDS
        ):
            self._auto_resume_done = True
            self._on_resume_live()

        frame = self.camera.get_frame() if self.mode == "live" else self.captured_frame
        if frame is not None:
            self._render_frame(frame)

        self._update_readouts()
        self._update_voice_readout()
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
        self._draw_auto_scan_countdown(frame)

        # On a finished capture, draw a result banner across the bottom.
        if self.mode == "frozen" and self.result and not self.analyzing:
            special = self._is_special_result(self.result)
            cat = self.result["category"]
            color = ATTENTION_RED_RGB if special else config.CATEGORY_COLOR.get(
                cat, config.NEUTRAL_COLOR)
            category_name = "SPECIAL HANDLING" if special else self.result["category_name"]
            caption = f"{self.result['object']}  ->  {category_name}"
            if self.result.get("confidence"):
                caption += f"  ({self.result['confidence']}%)"
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 46), (w, h), _rgb_to_bgr(color), -1)
            cv2.putText(frame, caption, (14, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # Fit the frame into the video panel by FILLING (no black bars). We
        # scale so the smaller dimension matches, then centre-crop the
        # overflow on the larger one. This keeps aspect ratio (no stretching)
        # and never leaves letterbox borders.
        h, w = frame.shape[:2]
        max_w, max_h = self._video_max_size()
        if max_w > 0 and max_h > 0:
            scale = max(max_w / w, max_h / h)            # FILL, not fit
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            if (new_w, new_h) != (w, h):
                frame = cv2.resize(frame, (new_w, new_h),
                                   interpolation=cv2.INTER_AREA
                                   if scale < 1.0 else cv2.INTER_LINEAR)
            # Centre-crop any overflow so the result exactly matches the panel.
            x_off = max(0, (new_w - max_w) // 2)
            y_off = max(0, (new_h - max_h) // 2)
            frame = frame[y_off:y_off + max_h, x_off:x_off + max_w]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=self.photo)

    def _video_max_size(self):
        self.root.update_idletasks()
        # Fall back to a comfy default until Tk has measured the label.
        width = self.video_label.winfo_width()
        height = self.video_label.winfo_height()
        if width <= 1 or height <= 1:
            return 640, 480
        return max(200, width), max(160, height)

    def _toggle_fullscreen(self):
        self._set_fullscreen(not self.fullscreen)

    def _set_fullscreen(self, enabled):
        self.fullscreen = enabled
        self.root.attributes("-fullscreen", enabled)

    def _set_background_theme(self, bg):
        """Recolour the page background. Skips the speech bubble (it has its
        own active colour) and the bin-pill canvases (their pills are repainted
        by _update_bins)."""
        self.root.configure(bg=bg)
        for widget in self.bg_widgets:
            try:
                widget.configure(bg=bg)
            except tk.TclError:
                pass
        # Each bin pill sits inside a canvas; make the canvas bg follow the
        # page so the pill's rounded edges blend in. The pill itself is
        # repainted by _update_bins on every tick.
        for cat, info in getattr(self, "bin_cards", {}).items():
            info["canvas"].configure(bg=bg)
            self._draw_bin_pill(cat)
        # The speech bubble's outer canvas also blends into the page so the
        # area around its rounded corners and tail doesn't show a cream sliver
        # during the red flash.
        bubble = getattr(self, "speech_bubble", None)
        if bubble is not None:
            bubble.configure(page_bg=bg)

    @staticmethod
    def _is_special_result(result):
        return bool(result and result.get("special"))

    def _attention_flash_bg(self):
        fresh = (time.monotonic() - self._result_time) < RESULT_HOLD_SECONDS
        if not fresh:
            return ATTENTION_RED
        return ATTENTION_RED if int(time.time() * 5) % 2 == 0 else THEME_BG

    def _next_focus(self, frame):
        local_box, local_label, local_ready, local_error = self.local_focus.snapshot()
        keep_previous_briefly = False
        if local_ready:
            candidate = local_box
            label = local_label
        elif local_error:
            candidate = detect_held_item_box(frame, self.focus_box)
            label = None
            keep_previous_briefly = True
        else:
            candidate = self.focus_box
            label = "loading"
            keep_previous_briefly = True

        if candidate is None:
            self.focus_misses += 1
            self._reset_auto_scan()
            if keep_previous_briefly and self.focus_misses <= FOCUS_MAX_MISSES:
                self.last_focus_box = self.focus_box
                return self.focus_box, None
            self.focus_box = None
            self.last_focus_box = None
            return None, None

        self.focus_misses = 0
        self.focus_box = _smooth_box(self.focus_box, candidate)
        self.last_focus_box = self.focus_box
        self._update_auto_scan_state(self.focus_box)
        return self.focus_box, label

    def _reset_auto_scan(self):
        self.still_reference_box = None
        self.still_started_at = None
        self.auto_scan_remaining = None

    def _update_auto_scan_state(self, box):
        if (
            not AUTO_SCAN_ENABLED
            or self.mode != "live"
            or self.analyzing
            or self.speaking
        ):
            self._reset_auto_scan()
            return
        if box is None:
            self._reset_auto_scan()
            return

        now = time.monotonic()
        if self.still_reference_box is None:
            self.still_reference_box = box
            self.still_started_at = now
            self.auto_scan_remaining = AUTO_SCAN_SECONDS
            return

        if _box_motion_ratio(self.still_reference_box, box) > AUTO_SCAN_MOTION_TOLERANCE:
            self.still_reference_box = box
            self.still_started_at = now
            self.auto_scan_remaining = AUTO_SCAN_SECONDS
            return

        elapsed = now - self.still_started_at
        self.auto_scan_remaining = max(0.0, AUTO_SCAN_SECONDS - elapsed)

    def _auto_scan_ready(self):
        return (
            AUTO_SCAN_ENABLED
            and self.auto_scan_remaining is not None
            and self.auto_scan_remaining <= 0
        )

    def _draw_auto_scan_countdown(self, frame):
        if (
            self.mode != "live"
            or self.auto_scan_remaining is None
            or not AUTO_SCAN_ENABLED
            or self.speaking
        ):
            return
        h, w = frame.shape[:2]
        remaining = max(0.0, self.auto_scan_remaining)
        text = f"Auto scan in {remaining:0.1f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(16, (w - tw) // 2)
        y = 46
        cv2.rectangle(frame, (x - 16, y - th - 16), (x + tw + 16, y + 14),
                      (30, 18, 55), -1)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)

    def _update_readouts(self):
        # Reset the detail readouts to a quiet state by default; specific modes
        # below fill them in.
        def quiet_details():
            self.object_label.config(text="", bg=THEME_BG, fg=THEME_SUBTLE)
            self.category_label.config(text="", bg=THEME_BG, fg=THEME_SUBTLE)
            self.confidence_label.config(text="", bg=THEME_BG, fg=THEME_SUBTLE)
            self.reason_label.config(text="", bg=THEME_BG, fg=THEME_SUBTLE)

        if self.mode == "live":
            self._set_background_theme(THEME_BG)
            if self.speaking:
                text = "Still speaking result..."
            elif self.auto_scan_remaining is not None and AUTO_SCAN_ENABLED:
                text = (f"Hold still!\n"
                        f"Auto-scan in {self.auto_scan_remaining:0.1f}s")
            else:
                text = self._tips_text
            self._set_speech(text)
            quiet_details()
            self._set_details_visible(False)
            # Voice hint is most useful while idle. Keep it visible whenever
            # we're not mid-speech.
            self._set_voice_hint_visible(not self.speaking)
            return

        # Anything past this point is "frozen" -- we have a result (or are
        # waiting for one), so hide the sub-hint to keep the eye on Oscar.
        self._set_voice_hint_visible(False)
        self._set_details_visible(True)

        if self.analyzing:
            self._set_background_theme(THEME_BG)
            self._set_speech("Analyzing...\nOscar is thinking...",
                             fill="#fff0c0", fg="#7a5a00",
                             border=SPEECH_BORDER)
            quiet_details()
            self.object_label.config(text="Item: ...")
            return

        r = self.result or {}
        if r.get("error"):
            self._set_background_theme(THEME_BG)
            self._set_speech("Couldn't classify",
                             fill="#ffd1d1", fg="#7a0000",
                             border="#cc4444")
            quiet_details()
            self.object_label.config(text=f"Error: {r['error']}")
            return

        cat = r.get("category")
        name = r.get("category_name", "Unsure")

        if self._is_special_result(r):
            bg = self._attention_flash_bg()
            self._set_background_theme(bg)
            self._set_speech("DO NOT THROW\nSPECIAL HANDLING",
                             fill=ATTENTION_RED, fg="#ffffff",
                             border="#ffffff")
            self.object_label.config(text=f"Item: {r.get('object', '-')}",
                                     bg=bg, fg="#ffffff")
            self.category_label.config(text="Bin: do not use these bins",
                                       bg=bg, fg="#ffffff")
            self.confidence_label.config(
                text=f"Confidence: {r.get('confidence', 0)}%",
                bg=bg, fg="#ffffff")
            self.reason_label.config(text=r.get("reason", ""),
                                     bg=bg, fg="#ffffff")
            return

        # Normal categorised result.
        self._set_background_theme(THEME_BG)
        if cat is not None:
            color = BIN_PILL_COLOR.get(
                cat, _rgb_to_hex(config.CATEGORY_COLOR.get(cat,
                                                           config.NEUTRAL_COLOR)))
            self._set_speech(f"DROP IN\n{name.upper()}",
                             fill=color, fg="#ffffff", border=color)
        else:
            self._set_speech("Unsure - try capturing again",
                             fill="#fff0c0", fg="#7a5a00",
                             border=SPEECH_BORDER)
        self.object_label.config(text=f"Item: {r.get('object', '-')}",
                                 bg=THEME_BG, fg=THEME_TEXT)
        self.category_label.config(text=f"Bin: {name}",
                                   bg=THEME_BG, fg=THEME_TEXT)
        self.confidence_label.config(
            text=f"Confidence: {r.get('confidence', 0)}%",
            bg=THEME_BG, fg=THEME_SUBTLE)
        self.reason_label.config(text=r.get("reason", ""),
                                 bg=THEME_BG, fg=THEME_SUBTLE)

    def _update_voice_readout(self):
        if self.voice_status_label is None:
            return
        status, error, last_heard = self.voice_trigger.snapshot()
        text = status
        if error:
            text += f"\n{error}"
        elif last_heard:
            text += f'\nHeard: "{last_heard}"'
        self.voice_status_label.config(text=text)

    def _update_bins(self):
        """Repaint each bin pill. The chosen bin flashes between its colour
        and white; on special-handling results, ALL pills flash red/white."""
        target = None
        special = False
        if self.mode == "frozen" and self.result and not self.analyzing:
            target = self.result.get("category")
            special = self._is_special_result(self.result)
        fresh = (time.monotonic() - self._result_time) < RESULT_HOLD_SECONDS
        blink_on = int(time.time() * 4) % 2 == 0

        for cat, info in self.bin_cards.items():
            base = info["color"]
            if special:
                fill = "#ffffff" if fresh and blink_on else ATTENTION_RED
                text_color = ATTENTION_RED if fill == "#ffffff" else "#ffffff"
            elif cat == target:
                fill = "#ffffff" if fresh and blink_on else base
                text_color = base if fill == "#ffffff" else "#ffffff"
            else:
                fill = base
                text_color = "#ffffff"
            self._set_bin_pill(cat, fill, text_color)

    def _on_close(self):
        self.voice_trigger.stop()
        self.local_focus.stop()
        self.camera.stop()
        self.root.after(150, self.root.destroy)


def main():
    root = tk.Tk()
    LLMSorterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
