"""
llm_client.py
=============
A small wrapper around a vision LLM. Given one webcam frame, it asks the model
to (a) identify the item and (b) recommend which bin it belongs
in, then returns a tidy Python dict.

IMPORTANT: this is only ever called when YOU press the Capture button -- one
API request per capture, never per frame -- so token/usage cost stays tiny and
fully under your control.

Gemini setup (one time):
  1. Get a free API key from https://aistudio.google.com/app/apikey
  2. Install the SDK:   pip install -r requirements-llm.txt
  3. Put your key in the environment before running:
        export GEMINI_API_KEY="your-key-here"      (macOS/Linux)
        setx   GEMINI_API_KEY "your-key-here"       (Windows)

Ollama / local Gemma setup:
  1. Install Ollama and pull a vision-capable model, e.g.:
        ollama pull gemma3:27b
  2. Run this app with:
        export LLM_PROVIDER=ollama
        export OLLAMA_MODEL=gemma3:27b

The bin definitions (names/colours) are reused from waste_sorter/config.py.
"""

import json
import os
import re
import base64
import urllib.error
import urllib.request

import cv2

from waste_sorter import config


# Default Gemini model. "flash" is fast + cheap and plenty accurate for this.
# Override with the LLM_MODEL (or GEMINI_MODEL) env var, e.g. "gemini-2.5-pro".
DEFAULT_GEMINI_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("GEMINI_MODEL") \
    or "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL") or os.environ.get("LLM_MODEL") \
    or "gemma3:27b"
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

# Friendly bin names the model is allowed to choose from (plus "Unsure").
BIN_NAMES = [config.CATEGORY_DISPLAY[c]["name"] for c in config.CATEGORIES]
SPECIAL_CATEGORY_NAME = "Special handling"
SPECIAL_CATEGORY_ALIASES = {
    SPECIAL_CATEGORY_NAME.lower(),
    "special",
    "special waste",
    "hazardous waste",
    "hazardous",
    "e-waste",
    "ewaste",
    "do not throw",
    "do not trash",
}
SPECIAL_OBJECT_KEYWORDS = {
    "phone",
    "cell phone",
    "smartphone",
    "battery",
    "batteries",
    "laptop",
    "tablet",
    "electronic",
    "electronics",
    "vape",
    "charger",
    "power bank",
    "medicine",
    "medication",
    "chemical",
    "sharps",
    "needle",
    "syringe",
}

# Map a friendly name (e.g. "Recycling") back to our internal key (RECYCLING).
NAME_TO_KEY = {config.CATEGORY_DISPLAY[c]["name"]: c for c in config.CATEGORIES}

# The instruction we send alongside the image.
PROMPT = f"""Classify the main held item in the image. The image may be cropped from a detector box; focus on the central boxed/cropped item. Ignore hands, faces, and background.

Bins:
Recycling = clean bottles, cans, glass, metal, rigid plastic, clean paper/cardboard.
Compost = food scraps, napkins, compostable paper/containers.
Landfill = wrappers, foam, dirty/mixed trash, coffee cups, utensils.
Special handling = items that should NOT go in recycling, compost, or landfill, including phones, laptops, electronics, batteries, vapes, chargers, power banks, chemicals, medicine, sharps, or anything hazardous.
Use Unsure if unclear.

Return only JSON:
{{"object":"short item name","category":"{'|'.join(BIN_NAMES)}|{SPECIAL_CATEGORY_NAME}|Unsure","confidence":0-100,"reason":"short reason"}}"""


# ---------------------------------------------------------------------------
# Pure helper (no network) -- unit-tested separately.
# ---------------------------------------------------------------------------
def parse_response(text):
    """Turn the model's raw text answer into a normalised result dict.

    Returns dict with keys:
      object        (str)
      category      (str|None)  -> internal bin key, or None if Unsure/special
      category_name (str)       -> friendly name ("Recycling"/"Unsure"/special)
      confidence    (int 0-100)
      reason        (str)
    Raises ValueError if no JSON could be found at all.
    """
    if not text:
        raise ValueError("Empty response from the model")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse JSON from: {text!r}")
        data = json.loads(match.group(0))

    obj = str(data.get("object", "")).strip() or "unknown item"
    raw_cat = str(data.get("category", "")).strip()
    reason = str(data.get("reason", "")).strip()

    category_name = "Unsure"
    category_key = None
    special = False
    for name in BIN_NAMES:
        if raw_cat.lower() == name.lower():
            category_name = name
            category_key = NAME_TO_KEY[name]
            break
    else:
        raw_lower = raw_cat.lower()
        if raw_lower in SPECIAL_CATEGORY_ALIASES:
            category_name = SPECIAL_CATEGORY_NAME
            special = True

    object_lower = obj.lower()
    if any(keyword in object_lower for keyword in SPECIAL_OBJECT_KEYWORDS):
        category_name = SPECIAL_CATEGORY_NAME
        category_key = None
        special = True

    try:
        confidence = int(round(float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0
    confidence = max(0, min(100, confidence))

    return {
        "object": obj,
        "category": category_key,
        "category_name": category_name,
        "confidence": confidence,
        "reason": reason,
        "special": special,
    }


# ---------------------------------------------------------------------------
# The client (does the network call).
# ---------------------------------------------------------------------------
class LLMClassifier:
    def __init__(self, api_key=None, model=None):
        self.provider = DEFAULT_PROVIDER
        if self.provider in {"ollama", "gemma", "local"}:
            self.provider = "ollama"
            self.model = model or DEFAULT_OLLAMA_MODEL
            self.ollama_host = DEFAULT_OLLAMA_HOST.rstrip("/")
            self.client = None
            return

        if self.provider not in {"gemini", "google"}:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER={self.provider!r}. "
                'Use "gemini" or "ollama".')
        self.provider = "gemini"

        # Import here so the rest of the app can be imported without the SDK.
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The LLM SDK isn't installed. Run:\n"
                "    pip install -r requirements-llm.txt"
            ) from exc

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY") or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No API key found. Get one at "
                "https://aistudio.google.com/app/apikey then set it:\n"
                '    export GEMINI_API_KEY="your-key-here"')

        self.model = model or DEFAULT_GEMINI_MODEL
        self._genai = genai
        self.client = genai.Client(api_key=self.api_key)

    def classify(self, frame_bgr):
        """Send one BGR frame to the model and return a result dict.

        On any error, returns a dict with category=None and an 'error' field
        so the UI can show a friendly message instead of crashing.
        """
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return {"object": "", "category": None, "category_name": "Unsure",
                    "confidence": 0, "reason": "", "error": "Could not encode frame"}

        if self.provider == "ollama":
            return self._classify_ollama(buf.tobytes())
        return self._classify_gemini(buf.tobytes())

    def _classify_gemini(self, image_bytes):
        from google.genai import types

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[image_part, PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            result = parse_response(response.text)
            result["error"] = None
            return result
        except Exception as exc:  # noqa: BLE001
            return {"object": "", "category": None, "category_name": "Unsure",
                    "confidence": 0, "reason": "", "error": str(exc)}

    def _classify_ollama(self, image_bytes):
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "prompt": PROMPT,
            "images": [image_b64],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }
        request = urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read().decode("utf-8")
            data = json.loads(body)
            result = parse_response(data.get("response", ""))
            result["error"] = None
            return result
        except urllib.error.URLError as exc:
            return {"object": "", "category": None, "category_name": "Unsure",
                    "confidence": 0, "reason": "",
                    "error": f"Could not reach Ollama at {self.ollama_host}: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"object": "", "category": None, "category_name": "Unsure",
                    "confidence": 0, "reason": "", "error": str(exc)}
