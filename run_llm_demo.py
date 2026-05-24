"""
run_llm_demo.py
===============
Entry point for the "LLM mode" trash sorter (capture-on-demand).

This is the alternative to the fully-local detector (`run_demo.py`). It shows a
live preview and only contacts the vision LLM when you press the Capture button,
so it uses very few tokens while giving you the model's strong general vision.

One-time setup:
    pip install -r requirements-llm.txt
    export GEMINI_API_KEY="your-key-here"     # get one at aistudio.google.com

Then run:
    python run_llm_demo.py
"""

from llm_sorter.app import main

if __name__ == "__main__":
    main()
