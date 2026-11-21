"""Source register: every number used in the tool, where it came from, how far to trust it."""
from pathlib import Path

import pandas as pd

SOURCES_CSV = Path(__file__).resolve().parent.parent / "data" / "sources.csv"


def load_sources():
    return pd.read_csv(SOURCES_CSV)
