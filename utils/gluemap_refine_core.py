"""Compatibility import for historical analysis scripts. Use ffba modules."""
import sys
from ffba import api
sys.modules[__name__] = api
