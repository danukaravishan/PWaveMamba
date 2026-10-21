"""Shared command-line plumbing of the scripts (paths relative to the repository root)."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pwavemamba.config import DEFAULT_WINDOW, WINDOWS, segment_filename  # noqa: E402


def base_parser(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"),
                    help="folder with waveforms.hdf5, metadata.csv, segments/ and christchurch/")
    ap.add_argument("--split-csv", default=os.path.join(ROOT, "splits", "event_split.csv"),
                    help="event -> split table (released split by default)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"), help="output folder")
    ap.add_argument("--window", default=os.environ.get("PWAVEMAMBA_WINDOW", DEFAULT_WINDOW), choices=list(WINDOWS),
                    help="window variant: ID6 = 2 s + 2 s (main model); ID2/ID10/ID13/ID14 = 1/3/4/5 s after P")
    return ap


def segment_path(args):
    return os.path.join(args.data_dir, "segments", segment_filename(args.window))
