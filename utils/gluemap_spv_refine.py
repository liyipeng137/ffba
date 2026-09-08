"""Compatibility import for existing analysis scripts."""
import sys
from ffba import refinement
sys.modules[__name__] = refinement
