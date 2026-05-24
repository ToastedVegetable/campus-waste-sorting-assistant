"""
smoothing.py
============
Raw model output is jumpy: from frame to frame the predicted category can
flicker (bottle -> nothing -> bottle -> cup ...). Showing that directly looks
broken and makes the bins blink randomly.

This module fixes that with two simple ideas:

  1. TEMPORAL SMOOTHING (majority vote)
     Keep the last N detections. The "current" category is whichever one
     appears most often in that window, and we report the AVERAGE confidence
     of the votes for it. One bad frame can't change the answer on its own.

  2. A STABILITY TIMER
     We only "lock in" a category (and flash its bin) once it has stayed on
     top continuously for STABLE_SECONDS. This is the "hold the item steady
     for a moment, then it tells you where it goes" behaviour.

The class below has no knowledge of cameras or UI -- it just eats detections
and tells you the smoothed state. That makes it easy to unit-test.
"""

import time
from collections import deque, Counter

from . import config


# A small sentinel meaning "no object detected this tick".
NOTHING = "NOTHING"


class TemporalSmoother:
    def __init__(self,
                 window: int = None,
                 stable_seconds: float = None,
                 category_confidence: float = None):
        self.window = window or config.SMOOTHING_WINDOW
        self.stable_seconds = stable_seconds if stable_seconds is not None \
            else config.STABLE_SECONDS
        self.category_confidence = category_confidence if category_confidence is not None \
            else config.CATEGORY_CONFIDENCE

        # Recent votes: each entry is (category_or_NOTHING, confidence).
        self._votes = deque(maxlen=self.window)

        # Which category is currently "on top", and since when.
        self._current = NOTHING
        self._current_since = time.monotonic()

    def update(self, detection):
        """Feed one detection (or None) into the smoother.

        `detection` is a detector.Detection or None. Categories that are None
        (unmapped/unsure) are treated as NOTHING for voting purposes.
        """
        if detection is None or detection.category is None:
            self._votes.append((NOTHING, 0.0))
        else:
            self._votes.append((detection.category, detection.confidence))

        # Recompute which category currently wins the majority vote.
        winner = self._majority_category()
        if winner != self._current:
            # The leading category changed -> restart the stability timer.
            self._current = winner
            self._current_since = time.monotonic()

    def _majority_category(self):
        """Return the most common category in the window (may be NOTHING)."""
        if not self._votes:
            return NOTHING
        counts = Counter(cat for cat, _ in self._votes)
        # most_common(1) -> [(category, count)]
        return counts.most_common(1)[0][0]

    def _average_confidence(self, category):
        """Average confidence of the votes that picked `category`."""
        confs = [c for cat, c in self._votes if cat == category]
        if not confs:
            return 0.0
        return sum(confs) / len(confs)

    def state(self):
        """Return the smoothed state as a plain dictionary.

        Keys:
          category   : winning category key, or None if nothing/unsure
          confidence : averaged confidence (0.0 - 1.0)
          confident  : True if confidence >= CATEGORY_CONFIDENCE
          stable     : True if a confident category has held for STABLE_SECONDS
          progress   : 0.0 - 1.0 progress toward "stable" (good for a bar)
        """
        category = self._current
        if category == NOTHING:
            return {
                "category": None,
                "confidence": 0.0,
                "confident": False,
                "stable": False,
                "progress": 0.0,
            }

        confidence = self._average_confidence(category)
        confident = confidence >= self.category_confidence

        held_for = time.monotonic() - self._current_since
        # Only count toward stability while we're actually confident.
        progress = min(held_for / self.stable_seconds, 1.0) if confident else 0.0
        stable = confident and held_for >= self.stable_seconds

        return {
            "category": category,
            "confidence": confidence,
            "confident": confident,
            "stable": stable,
            "progress": progress,
        }
