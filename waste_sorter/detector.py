"""
detector.py
===========
A thin, friendly wrapper around a LOCAL YOLOv8 object-detection model.

Why this file exists:
  * It hides the Ultralytics/YOLO details behind one small class.
  * It returns a simple `Detection` object (label, confidence, box, category)
    so the rest of the demo never has to know about tensors or model output.
  * Everything runs on YOUR machine. No frames ever leave the laptop, and no
    LLM / cloud API is called.

Swapping in your own trained model later:
  * Point config.MODEL_WEIGHTS at your own .pt file.
  * Update config.LABEL_TO_CATEGORY so YOUR class names map to the four bins.
  * Nothing in this file needs to change.
"""

from dataclasses import dataclass

from . import config


@dataclass
class Detection:
    """One detected object, already translated into our waste vocabulary."""
    label: str           # raw model label, e.g. "bottle"
    confidence: float    # 0.0 - 1.0
    box: tuple           # (x1, y1, x2, y2) pixel coordinates in the frame
    category: str        # one of config.CATEGORIES, or None if unmapped/unsure


class WasteDetector:
    """Loads a YOLOv8 model once and runs detection on individual frames."""

    def __init__(self, weights: str = None, conf: float = None,
                 imgsz: int = None, device: str = None):
        # Import here (not at top of file) so that simply importing this module
        # does not require the heavy `ultralytics` package to be installed --
        # handy for reading/testing the other modules.
        from ultralytics import YOLO

        self.weights = weights or config.MODEL_WEIGHTS
        self.conf = conf if conf is not None else config.DETECTION_CONFIDENCE
        self.imgsz = imgsz if imgsz is not None else config.INFERENCE_IMGSZ
        self.device = device if device is not None else (
            config.DEVICE or self._auto_device())

        print(f"[detector] Loading model '{self.weights}' on device "
              f"'{self.device}' (imgsz={self.imgsz}) ...")
        # On first run this downloads the weights file, then caches it.
        self.model = YOLO(self.weights)
        # `model.names` maps numeric class ids -> human label strings.
        self.names = self.model.names
        print("[detector] Model ready.")

    @staticmethod
    def _auto_device():
        """Pick the fastest available device: Apple GPU > NVIDIA GPU > CPU."""
        try:
            import torch
            if torch.backends.mps.is_available():   # Apple Silicon (Metal)
                return "mps"
            if torch.cuda.is_available():           # NVIDIA GPU
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def detect(self, frame):
        """Run the model on a single BGR frame (as captured by OpenCV).

        Returns the SINGLE most relevant detection (highest confidence among
        objects we can map to a bin), or None if nothing confident was found.

        We deliberately pick just one object to keep the demo's UX simple:
        the user holds up one item at a time.
        """
        # verbose=False keeps the console quiet. conf= applies the threshold.
        # device/imgsz make inference run on the GPU at higher resolution.
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                     device=self.device, verbose=False)
        if not results:
            return None

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_w * frame_h)
        cx_frame, cy_frame = frame_w / 2.0, frame_h / 2.0
        # Half the diagonal -- used to normalise how far a box is from center.
        max_dist = (cx_frame ** 2 + cy_frame ** 2) ** 0.5

        best = None         # the Detection we will return
        best_score = -1.0   # its selection score (higher = better)

        # Loop over every box the model produced for this frame.
        for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.names.get(class_id, str(class_id)) \
                if isinstance(self.names, dict) else self.names[class_id]

            # Skip anything on the ignore list (e.g. "person") so the demo
            # focuses on the item being held up, not the user or background.
            if label in config.IGNORED_LABELS:
                continue

            # Pixel coordinates of the box corners.
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

            # --- Filter out tiny boxes (usually clutter / false positives) ---
            area_frac = ((x2 - x1) * (y2 - y1)) / frame_area
            if area_frac < config.MIN_BOX_AREA_FRACTION:
                continue

            # --- Score the box so we pick the item most likely being shown ---
            # A held-up item is typically large and roughly centered, so we
            # blend confidence with size and centrality instead of trusting
            # raw confidence alone. This cuts down on the model latching onto
            # small background objects.
            box_cx, box_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = ((box_cx - cx_frame) ** 2 + (box_cy - cy_frame) ** 2) ** 0.5
            centrality = 1.0 - min(dist / max_dist, 1.0)   # 1=center, 0=corner
            size_score = min(area_frac * 4.0, 1.0)          # caps at 25% of frame
            score = confidence * (0.6 + 0.25 * centrality + 0.15 * size_score)

            if score > best_score:
                best_score = score
                best = Detection(
                    label=label,
                    confidence=confidence,
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    category=config.category_for_label(label),
                )

        return best
