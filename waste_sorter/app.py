"""
app.py
======
The Tkinter user interface and the glue that ties the camera, the detector,
and the smoother together.

Big-picture flow
----------------
  [Camera thread]                         [Tkinter UI thread]
  grab webcam frame  ---- shared state --->  read latest frame
  every Nth frame:                           draw bounding box
    run YOLO detect  ---- shared state --->  feed smoother (on new result)
                                             update labels + bin cards
                                             ask Tk to call us again in ~15ms

Why a separate camera thread?
  Running the YOLO model can take ~50-150 ms per frame on a laptop CPU. If we
  did that inside Tkinter's event loop, the whole window would freeze each
  time. By doing capture + detection on a background thread and only *reading*
  the latest result in the UI thread, the video stays smooth.
"""

import threading
import time

import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

from . import config
from .detector import WasteDetector
from .smoothing import TemporalSmoother


# Largest size (pixels) we will display the video at, preserving aspect ratio.
DISPLAY_MAX_W = 760
DISPLAY_MAX_H = 560


def _rgb_to_hex(rgb):
    """(r, g, b) ints -> '#rrggbb' for Tkinter."""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgb_to_bgr(rgb):
    """OpenCV draws in BGR order; our config stores RGB."""
    r, g, b = rgb
    return (b, g, r)


# ===========================================================================
# Background camera + detection thread
# ===========================================================================
class CameraThread(threading.Thread):
    """Continuously grabs frames and (every Nth frame) runs detection.

    The UI thread reads `latest_frame` / `latest_detection` under `lock`.
    `detection_seq` increments every time a NEW detection result is produced,
    so the UI knows when to feed the smoother exactly once per result.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()

        self.latest_frame = None        # most recent BGR frame (numpy array)
        self.latest_detection = None    # most recent Detection or None
        self.detection_seq = 0          # bumps on each new detection result

        self.model_ready = False
        self.error = None               # human-readable error string, if any
        self._running = True

    def run(self):
        # 1) Load the model (downloads ~6 MB weights on first ever run).
        try:
            detector = WasteDetector()
        except Exception as exc:  # noqa: BLE001 (we want to surface anything)
            self.error = (f"Failed to load the detection model.\n{exc}\n\n"
                          f"Did you run: pip install -r requirements.txt ?")
            return
        self.model_ready = True

        # 2) Open the webcam.
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            self.error = (f"Could not open webcam (index {config.CAMERA_INDEX}).\n"
                          f"Close other apps using the camera, or try a "
                          f"different CAMERA_INDEX in config.py.")
            return

        # 3) Main capture loop.
        frame_i = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Mirror horizontally so moving left feels left (like a mirror).
            frame = cv2.flip(frame, 1)

            # Only run the (expensive) model every Nth frame.
            if frame_i % config.PROCESS_EVERY_N_FRAMES == 0:
                try:
                    detection = detector.detect(frame)
                except Exception as exc:  # keep the app alive on a bad frame
                    detection = None
                    print(f"[camera] detection error: {exc}")
                with self.lock:
                    self.latest_detection = detection
                    self.detection_seq += 1

            with self.lock:
                self.latest_frame = frame
            frame_i += 1

        cap.release()

    def snapshot(self):
        """Thread-safe read of the current state for the UI."""
        with self.lock:
            return self.latest_frame, self.latest_detection, self.detection_seq

    def stop(self):
        self._running = False


# ===========================================================================
# The Tkinter application
# ===========================================================================
class TrashSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Campus Waste Sorting Assistant")
        self.root.configure(bg=config.UI_BG)

        self.smoother = TemporalSmoother()
        self.camera = CameraThread()
        self.last_seen_seq = -1
        self.photo = None  # keep a reference so the image isn't garbage-collected

        self._build_ui()

        # Start grabbing frames in the background, then begin the UI loop.
        self.camera.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._tick)

    # ---------------------------------------------------------------- UI build
    def _build_ui(self):
        # Title bar
        header = tk.Label(self.root, text="♻  Campus Waste Sorting Assistant",
                          font=("Helvetica", 22, "bold"),
                          bg=config.UI_BG, fg=config.UI_TEXT)
        header.grid(row=0, column=0, columnspan=2, pady=(14, 4))

        subtitle = tk.Label(self.root,
                            text="Hold one item up to the camera and keep it steady.",
                            font=("Helvetica", 12),
                            bg=config.UI_BG, fg=config.UI_SUBTLE_TEXT)
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # --- Left column: live video ---
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=2, column=0, padx=16, pady=8)

        # --- Right column: read-outs ---
        info = tk.Frame(self.root, bg=config.UI_BG)
        info.grid(row=2, column=1, padx=16, pady=8, sticky="n")

        self.status_label = tk.Label(info, text="Starting up…",
                                     font=("Helvetica", 17, "bold"),
                                     bg=config.UI_BG, fg=config.UI_TEXT,
                                     wraplength=300, justify="left")
        self.status_label.pack(anchor="w", pady=(0, 12))

        self.object_label = tk.Label(info, text="Detected: —",
                                     font=("Helvetica", 13),
                                     bg=config.UI_BG, fg=config.UI_SUBTLE_TEXT)
        self.object_label.pack(anchor="w")

        self.category_label = tk.Label(info, text="Category: —",
                                       font=("Helvetica", 13),
                                       bg=config.UI_BG, fg=config.UI_SUBTLE_TEXT)
        self.category_label.pack(anchor="w")

        self.confidence_label = tk.Label(info, text="Confidence: —",
                                         font=("Helvetica", 13),
                                         bg=config.UI_BG, fg=config.UI_SUBTLE_TEXT)
        self.confidence_label.pack(anchor="w", pady=(0, 12))

        # "Hold steady" progress toward locking a category in.
        tk.Label(info, text="Stability", font=("Helvetica", 10),
                 bg=config.UI_BG, fg=config.UI_SUBTLE_TEXT).pack(anchor="w")
        self.progress = ttk.Progressbar(info, length=300, maximum=1.0)
        self.progress.pack(anchor="w", pady=(0, 4))

        # --- Bottom row: the four bin cards ---
        bins = tk.Frame(self.root, bg=config.UI_BG)
        bins.grid(row=3, column=0, columnspan=2, pady=(8, 18))

        self.bin_cards = {}   # category key -> dict of widgets
        for i, cat in enumerate(config.CATEGORIES):
            disp = config.CATEGORY_DISPLAY[cat]
            card = tk.Frame(bins, bg=config.UI_PANEL_BG, width=190, height=110,
                            highlightthickness=3,
                            highlightbackground=config.UI_PANEL_BG)
            card.grid(row=0, column=i, padx=10)
            card.grid_propagate(False)  # keep a fixed card size

            name = tk.Label(card, text=disp["name"],
                            font=("Helvetica", 15, "bold"),
                            bg=config.UI_PANEL_BG, fg=config.UI_TEXT)
            name.pack(pady=(18, 2))

            hint = tk.Label(card, text=disp["hint"],
                            font=("Helvetica", 9), wraplength=170,
                            bg=config.UI_PANEL_BG, fg=config.UI_SUBTLE_TEXT)
            hint.pack()

            self.bin_cards[cat] = {"frame": card, "name": name, "hint": hint}

    # ------------------------------------------------------------- main loop
    def _tick(self):
        """Called ~every 15 ms by Tkinter. Reads state, redraws everything."""
        # Surface any fatal error from the camera thread.
        if self.camera.error:
            self.status_label.config(text=self.camera.error, fg="#ff6b6b")
            self.root.after(200, self._tick)
            return

        if not self.camera.model_ready:
            self.status_label.config(text="Loading detection model…\n(first run downloads ~6 MB)",
                                     fg=config.UI_TEXT)
            self.root.after(100, self._tick)
            return

        frame, detection, seq = self.camera.snapshot()

        # Feed the smoother exactly once per NEW detection result.
        if seq != self.last_seen_seq:
            self.smoother.update(detection)
            self.last_seen_seq = seq

        state = self.smoother.state()

        if frame is not None:
            self._render_frame(frame, detection, state)

        self._update_readouts(detection, state)
        self._update_bins(state)

        # ~60 fps UI refresh. The video stays smooth; detection runs slower.
        self.root.after(15, self._tick)

    # --------------------------------------------------------- drawing helpers
    def _render_frame(self, frame, detection, state):
        """Draw the bounding box + label onto the frame and show it."""
        frame = frame.copy()

        # Decide the box colour from the SMOOTHED category (steadier than the
        # raw per-frame detection), falling back to neutral grey.
        smoothed_cat = state["category"]
        color_rgb = config.CATEGORY_COLOR.get(smoothed_cat, config.NEUTRAL_COLOR)
        color_bgr = _rgb_to_bgr(color_rgb)

        # Only draw a box if we currently have a raw detection to anchor it to.
        if detection is not None:
            x1, y1, x2, y2 = detection.box
            thickness = 4 if state["stable"] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)

            # Build the caption shown above the box. This describes WHAT the
            # model thinks the object is (its detected label), not which bin
            # it goes in -- the bin guidance is shown in the side panel and
            # status line, and via the flashing bin card.
            caption = f"{detection.label}  {detection.confidence * 100:.0f}%"

            # A filled strip behind the text so it stays readable.
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ty = max(y1 - 10, th + 10)
            cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 10, ty + 4),
                          color_bgr, -1)
            cv2.putText(frame, caption, (x1 + 5, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # Resize to fit the display area while preserving aspect ratio.
        h, w = frame.shape[:2]
        scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # OpenCV is BGR; PIL/Tk want RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image)  # keep reference!
        self.video_label.config(image=self.photo)

    def _update_readouts(self, detection, state):
        """Update the text labels + progress bar + status message."""
        # Detected raw object label.
        if detection is not None:
            self.object_label.config(text=f"Detected: {detection.label}")
        else:
            self.object_label.config(text="Detected: —")

        # Smoothed category + confidence.
        cat = state["category"]
        if cat is not None and state["confident"]:
            cat_name = config.CATEGORY_DISPLAY[cat]["name"]
            self.category_label.config(text=f"Category: {cat_name}")
            self.confidence_label.config(text=f"Confidence: {state['confidence'] * 100:.0f}%")
        else:
            self.category_label.config(text="Category: —")
            self.confidence_label.config(text="Confidence: —")

        self.progress["value"] = state["progress"]

        # Status message (the headline that tells the user what to do).
        if state["stable"] and cat is not None:
            cat_name = config.CATEGORY_DISPLAY[cat]["name"]
            self.status_label.config(text=f"➜  Place in {cat_name}!", fg="#7CFC9A")
        elif state["confident"] and cat is not None:
            cat_name = config.CATEGORY_DISPLAY[cat]["name"]
            self.status_label.config(text=f"Looks like {cat_name}… hold steady",
                                     fg=config.UI_TEXT)
        elif detection is not None:
            # Something is there but we're not confident enough.
            self.status_label.config(text="Unsure — hold item steady",
                                     fg="#ffd166")
        else:
            self.status_label.config(text="Hold an item up to the camera",
                                     fg=config.UI_SUBTLE_TEXT)

    def _update_bins(self, state):
        """Highlight / flash the bin that matches the current prediction."""
        cat = state["category"]
        stable = state["stable"]
        confident = state["confident"]

        # A simple time-based blink: toggles a few times per second.
        blink_on = int(time.time() * 3) % 2 == 0

        for c, widgets in self.bin_cards.items():
            frame = widgets["frame"]

            is_target = (c == cat and confident)
            if is_target and stable:
                # Locked in: flash brightly between the bin colour and white.
                base = config.CATEGORY_COLOR[c]
                border = _rgb_to_hex(base) if blink_on else "#ffffff"
                fill = _rgb_to_hex(base)
                self._set_card_colors(widgets, fill, border)
            elif is_target:
                # Candidate (confident but not yet locked): steady soft outline.
                self._set_card_colors(widgets, config.UI_PANEL_BG,
                                      _rgb_to_hex(config.CATEGORY_COLOR[c]))
            else:
                # Idle.
                self._set_card_colors(widgets, config.UI_PANEL_BG,
                                      config.UI_PANEL_BG)

    @staticmethod
    def _set_card_colors(widgets, fill_hex, border_hex):
        widgets["frame"].config(bg=fill_hex, highlightbackground=border_hex,
                                highlightcolor=border_hex)
        widgets["name"].config(bg=fill_hex)
        widgets["hint"].config(bg=fill_hex)

    # ------------------------------------------------------------------ close
    def _on_close(self):
        self.camera.stop()
        # Give the thread a moment to release the camera before the window dies.
        self.root.after(150, self.root.destroy)


def main():
    root = tk.Tk()
    TrashSorterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
