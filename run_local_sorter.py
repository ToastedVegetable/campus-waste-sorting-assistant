"""
run_local_sorter.py
===================
Entry point for the local Campus Waste Sorting Assistant.

This opens a window showing your webcam. Hold an item (bottle, book, phone,
piece of fruit, ...) up to the camera and keep it steady for about 2 seconds;
the matching waste bin will light up and flash.

Press the window's close button (or Ctrl-C in the terminal) to quit.
"""

from waste_sorter.app import main

if __name__ == "__main__":
    main()
