"""
run_llm_demo.py
===============
Compatibility entry point for the LLM-assisted Campus Waste Sorting Assistant.

The preferred command is:

    python run_llm_sorter.py

This wrapper remains so older notes that use `python run_llm_demo.py` still
work.
"""

from run_llm_sorter import main

if __name__ == "__main__":
    main()
