# Campus Waste Sorting Assistant — LLM Mode

An alternative to the fully-local detector in `waste_sorter/`. Instead of
running a model on every frame, this app shows a **live preview** with a local
highlight around the item you are holding and waits for you to press
**Capture**. Only then does it send one clean, unannotated frame to a vision
**LLM**. It can use either Google's Gemini API or a local Ollama model, which
identifies the item and recommends Recycling, Compost, or Landfill.

This keeps the original local app completely untouched — it's a separate program.

## Why use this mode

- **Much stronger vision.** A large general-purpose model handles unusual or
  campus-specific items far better than a small local detector.
- **Tiny, controlled cost.** It calls the API **once per Capture press**, never
  per frame, so you decide exactly when a (cheap) request happens.

Trade-offs vs. the local app: Gemini needs internet + an API key, while Ollama
runs locally but is usually slower per capture. Either way, the app only calls
the LLM once when you press Capture.

## Setup (one time)

### Gemini

1. Get a free API key: https://aistudio.google.com/app/apikey
2. Install dependencies:

   ```bash
   pip install -r requirements-llm.txt
   ```

3. Set your key in the environment:

   ```bash
   export GEMINI_API_KEY="your-key-here"      # macOS/Linux
   # setx GEMINI_API_KEY "your-key-here"       # Windows (open a new terminal after)
   ```

### Local Ollama / Gemma

If you installed Ollama and pulled `gemma3:27b`, use:

```bash
export LLM_PROVIDER="ollama"
export OLLAMA_MODEL="gemma3:27b"
python run_llm_sorter.py
```

If Ollama is not already running, start it first:

```bash
ollama serve
```

## Run

```bash
python run_llm_sorter.py
```

Hold up one item, press **Capture & Classify**, and the model names the item,
picks a bin, shows its confidence and a one-line reason, and the matching bin
section flashes. The interface returns to live preview automatically after a
few seconds. The live highlight is only drawn in the interface; Gemini receives
a clean crop around the item selected by the local YOLO focus box, without the
overlay graphics. If the item is hazardous or should not go in any of the
three bins, such as a phone, battery, vape, or medicine, the app flashes red and
shows **Special handling** instead of choosing a bin.

After a classification, the app also speaks the recommendation out loud, for
example: "Throw in landfill, because it is a dirty wrapper." On macOS this uses
the built-in `say` command, so there is nothing extra to install.

If the item stays still in the live highlight for 5 seconds, the app also
captures automatically. The countdown appears on the camera feed and resets if
the item moves away.

You can also trigger capture by saying:

```text
hey oscar
oscar help
oscar scan
```

Hold the item up while saying one of those phrases. The app keeps a continuous
microphone stream open, checks a rolling audio buffer for the trigger phrases,
and uses SpeechRecognition's Google Web Speech recognizer. To turn this off:

```bash
export VOICE_TRIGGER="0"
python run_llm_sorter.py
```

## Customise

- **Fullscreen:** press `F11`, or start with `export FULLSCREEN=1`. Press
  `Esc` to leave fullscreen.
- **Theme:** the purple kiosk colours and local highlight search area
  (`ROI_W_FRAC` / `ROI_H_FRAC`) are constants at the top of `llm_sorter/app.py`.
- **Prompt / bin rules:** edit `PROMPT` in `llm_sorter/llm_client.py` to match
  your campus's exact recycling rules.
- **LLM crop:** by default, Gemini receives the clean crop around YOLO's focus
  box. Disable with `LLM_USE_FOCUS_CROP=0`, or tune padding with
  `LLM_CROP_PADDING_FRAC`.
- **Provider:** defaults to Gemini. Use `export LLM_PROVIDER="ollama"` for local
  Gemma through Ollama.
- **Model:** Gemini defaults to `gemini-2.5-flash`; Ollama defaults to
  `gemma3:27b`. Override with `GEMINI_MODEL`, `OLLAMA_MODEL`, or `LLM_MODEL`.
- **Voice trigger:** defaults to `hey oscar`, `oscar help`, `oscar scan`, and
  several close variants. Override with comma-separated `VOICE_TRIGGER_PHRASES`,
  tune the rolling buffer with `VOICE_WINDOW_SECONDS` / `VOICE_CHECK_INTERVAL`,
  or disable with `VOICE_TRIGGER=0`. The trigger also accepts close speech
  recognition misses that pair an Oscar-like word with an intent like "scan" or
  "help".
- **Text to speech:** enabled by default. Disable with `TEXT_TO_SPEECH=0`, or
  adjust macOS speech with `TTS_RATE` and `TTS_VOICE`.
- **Auto scan:** defaults to 5 seconds of holding still. Tune with
  `AUTO_SCAN_SECONDS`, adjust movement tolerance with `AUTO_SCAN_MOTION_TOLERANCE`,
  or disable with `AUTO_SCAN=0`.
- **Preview focus detector:** the live overlay uses the local YOLO detector on
  CPU by default to avoid Apple MPS crashes in the LLM app. Override with
  `FOCUS_DEVICE` or `FOCUS_IMGSZ` if you want to experiment.
- **Bins:** names/colours are shared with the local app via
  `waste_sorter/config.py`, so both apps stay consistent.

## Privacy note

When you press Capture, that single clean still image is sent to the selected
model for classification. With Gemini, it is uploaded to Google's API. With
Ollama, it stays on your laptop. The preview highlight graphics are local-only,
and nothing is sent while you're just previewing.

If the voice trigger is enabled, rolling audio snippets are sent to Google's web
speech recognizer to detect the trigger phrase.
