"""Campus Waste Sorting Assistant package.

A local, webcam-based computer-vision program that detects an object
and suggests which of three waste bins it should go in:
Recycling, Compost, or Landfill.

Modules
-------
config     : all tunable settings + the label -> waste-category mapping
detector   : thin wrapper around a local YOLOv8 model
smoothing  : temporal smoothing + a stability timer (stops the flicker)
app        : the Tkinter UI and the main loop that ties everything together
"""
