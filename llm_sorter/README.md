# LLM mode trash sorter (capture-on-demand)

An alternative to the fully-local detector in `waste_sorter/`. Instead of
running a model on every frame, this app shows a **live preview** with a local
highlight around the item you are holding and waits for you to press
**Capture**. Only then does it send one clean, unannotated frame to a vision
**LLM**. It can use either Google's Gemini API or a local Ollama model such as
`gemma3:27b`, which identifies the item and recommends a bin.

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
python run_llm_demo.py
```

If Ollama is not already running, start it first:

```bash
ollama serve
```

## Run

```bash
python run_llm_demo.py
```

Hold up one item, press **Capture & Classify**, and the model names the item,
picks a bin, shows its confidence and a one-line reason, and the matching bin
card lights up. Press **Resume Live** (or Capture again) to return to the
preview. The live highlight is only drawn in the interface; Gemini receives the
clean camera frame without the highlight box.

## Customise

- **Theme:** the Northwestern-purple colours and local highlight search area
  (`ROI_W_FRAC` / `ROI_H_FRAC`) are constants at the top of `llm_sorter/app.py`.
- **Prompt / bin rules:** edit `PROMPT` in `llm_sorter/llm_client.py` to match
  your campus's exact recycling rules.
- **Provider:** defaults to Gemini. Use `export LLM_PROVIDER="ollama"` for local
  Gemma through Ollama.
- **Model:** Gemini defaults to `gemini-2.5-flash`; Ollama defaults to
  `gemma3:27b`. Override with `GEMINI_MODEL`, `OLLAMA_MODEL`, or `LLM_MODEL`.
- **Preview focus detector:** the live overlay uses the local YOLO detector on
  CPU by default to avoid Apple MPS crashes in the LLM app. Override with
  `FOCUS_DEVICE` or `FOCUS_IMGSZ` if you want to experiment.
- **Bins:** names/colours are shared with the local app via
  `waste_sorter/config.py`, so both apps stay consistent.

## Privacy note

When you press Capture, that single clean still image is sent to the selected
model for classification. With Gemini, it is uploaded to Google's API. With
Ollama, it stays on your laptop. The preview highlight is local-only, and
nothing is sent while you're just previewing.
