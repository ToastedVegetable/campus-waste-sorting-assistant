"""LLM-mode trash sorter (capture-on-demand variant).

A SEPARATE app from the live, fully-local detector in `waste_sorter/`. Instead
of running a model on every frame, it shows a live preview and waits for you to
press a "Capture" button. Only then does it send that ONE frame to a vision LLM
(Google's Gemini, under the hood), which identifies the item and recommends a
bin.

Why this design:
  * The LLM's general vision is far stronger than a small local model, so
    accuracy on unusual/campus items is excellent.
  * Because it only calls the API once per button press (never per frame),
    it uses minimal tokens -- you control exactly when a request happens.

Files
-----
llm_client : talks to the vision LLM and parses its answer
app        : the Tkinter UI (live preview + Capture button + bin cards)
"""
