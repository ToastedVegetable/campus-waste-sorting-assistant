"""
run_demo.py
===========
Entry point for the Trash Sorter demo. Just run:

    python run_demo.py

This opens a window showing your webcam. Hold an item (bottle, book, phone,
piece of fruit, ...) up to the camera and keep it steady for ~2 seconds; the
matching waste bin will light up and flash.

Press the window's close button (or Ctrl-C in the terminal) to quit.
"""

from waste_sorter.app import main

if __name__ == "__main__":
    main()
