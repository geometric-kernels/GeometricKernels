"""Adds __version__"""

from pathlib import Path

with open(str(Path(__file__).parent.parent / "VERSION"), "r") as file:
    __version__ = file.read().strip()