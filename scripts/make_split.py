"""Event-grouped, magnitude-stratified 80/10/10 split.

    python scripts/make_split.py --metadata data/metadata.csv --out-dir outputs/splits

* Unit of splitting = GeoNet event id (trace-name prefix, e.g. 2016p858000): all records of an event,
  and all P, S and noise windows of those records, land in the same split (event-disjoint).
* Stratified by event magnitude bins [3,4), [4,5), [5,6), [6,inf) so that the magnitude distribution
  is similar in training, validation and test.
* Deterministic and machine-independent: events are ordered by sha256(seed + event id) inside each
  bin and cut at 80% / 90%. No random number generator is used, so the split is the same for every
  window length.
The released split is splits/event_split.csv. Needs only metadata.csv (columns trace_name, magnitude);
standard library only.
"""
import argparse
import collections
import csv
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--metadata", default=os.path.join(ROOT, "data", "metadata.csv"))
ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "splits"))
ap.add_argument("--seed", default="tgrs-2026")
a = ap.parse_args()

BINS = [(3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 99.0)]
os.makedirs(a.out_dir, exist_ok=True)

mag, nrec = {}, collections.Counter()
with open(a.metadata) as f:
    for r in csv.DictReader(f):
        ev = r["trace_name"].split("_")[0]
        mag[ev] = float(r["magnitude"])
        nrec[ev] += 1


def bin_of(m):
    return next(i for i, (lo, hi) in enumerate(BINS) if lo <= m < hi)


split = {}
for b in range(len(BINS)):
    evs = sorted((e for e in mag if bin_of(mag[e]) == b),
                 key=lambda e: hashlib.sha256((a.seed + e).encode()).hexdigest())
    n = len(evs)
    # validation and test each get round(10 %) of the bin, at least 1 (bins with >= 3 events)
    n_va = n_te = max(1, int(round(0.1 * n))) if n >= 3 else 0
    n_tr = n - n_va - n_te
    for i, e in enumerate(evs):
        split[e] = "train" if i < n_tr else ("validation" if i < n_tr + n_va else "test")

with open(os.path.join(a.out_dir, "event_split.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["event_id", "magnitude", "n_records", "split"])
    for e in sorted(split):
        w.writerow([e, f"{mag[e]:.3f}", nrec[e], split[e]])

summ = {"rule": "event-grouped, magnitude-stratified 80/10/10, sha256 order, seed " + a.seed, "per_split": {}}
for s in ("train", "validation", "test"):
    E = [e for e in split if split[e] == s]
    m = [mag[e] for e in E]
    summ["per_split"][s] = {
        "events": len(E), "records": sum(nrec[e] for e in E),
        "events_by_bin": {f"{lo:g}-{hi:g}": sum(1 for x in m if lo <= x < hi) for lo, hi in BINS},
        "records_by_bin": {f"{lo:g}-{hi:g}": sum(nrec[e] for e in E if lo <= mag[e] < hi) for lo, hi in BINS},
        "max_magnitude": max(m), "mean_magnitude": sum(m) / len(m)}
tot = sum(nrec.values())
for s in summ["per_split"]:
    summ["per_split"][s]["record_fraction"] = summ["per_split"][s]["records"] / tot
with open(os.path.join(a.out_dir, "event_split_summary.json"), "w") as f:
    json.dump(summ, f, indent=2)
print(json.dumps(summ, indent=1))
