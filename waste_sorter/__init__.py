"""Trash Sorter demo package.

A small, local, webcam-based computer-vision demo that detects an object
and suggests which of four waste bins it should go in:
Landfill, Paper, Recycling, or Special Waste.

Modules
-------
config     : all tunable settings + the label -> waste-category mapping
detector   : thin wrapper around a local YOLOv8 model
smoothing  : temporal smoothing + a stability timer (stops the flicker)
app        : the Tkinter UI and the main loop that ties everything together
"""
