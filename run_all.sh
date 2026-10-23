#!/usr/bin/env bash
# Full reproduction of the paper's experiments on a GPU machine (mamba_ssm installed).
# Run from the repository root:   JOBS=3 bash run_all.sh 2>&1 | tee outputs/run_all.log
# Needs data/waveforms.hdf5, data/metadata.csv and (for the case study) data/christchurch/ -- see docs/DATA.md.
# JOBS = trainings in parallel on one GPU (the model is small; 3 fit on an H100). Finished steps are skipped.
set -uo pipefail
JOBS="${JOBS:-3}"
OUT=outputs
mkdir -p "$OUT"/{models,train,eval,christchurch,bench}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 1. segment files (4 s main window + window-length study)
for W in ID6 ID2 ID10 ID13 ID14; do
  f=$(python -c "from pwavemamba.config import segment_filename as s; print(s('$W'))")
  [[ -f data/segments/$f ]] || { log "extract $W"; python scripts/extract_segments.py --window $W || exit 1; }
done

# 2. split: the released splits/event_split.csv is used. To check that it is reproduced from your metadata:
python scripts/make_split.py --out-dir "$OUT/splits" > /dev/null && \
  { cmp -s "$OUT/splits/event_split.csv" splits/event_split.csv && log "split reproduced" || log "WARNING: split differs from splits/event_split.csv"; }

# 3. training (main model first)
RUNS=("base 42 ID6" "no_pcond 42 ID6" "lstm 42 ID6" "transformer 42 ID6" "det_only 42 ID6" "int_only 42 ID6"
      "mag_only 42 ID6" "base 7 ID6" "base 123 ID6" "mamba_l2 42 ID6"
      "lstm 7 ID6" "lstm 123 ID6" "transformer 7 ID6" "transformer 123 ID6"
      "base 42 ID2" "base 42 ID10" "base 42 ID13" "base 42 ID14")
tagof() { [[ $3 == ID6 ]] && echo "${1}_s${2}" || echo "${1}_s${2}_${3}"; }
train_one() { set -- $1; t=$(tagof "$@"); [[ -f outputs/models/$t.pt ]] && return 0
  python scripts/train.py --variant "$1" --seed "$2" --window "$3" > "outputs/train/$t.log" 2>&1 \
    && echo "done $t" || echo "FAILED $t (see outputs/train/$t.log)"; }
export -f train_one tagof
log "training ${#RUNS[@]} runs, $JOBS in parallel"
printf '%s\n' "${RUNS[@]}" | xargs -P "$JOBS" -I{} bash -c 'train_one "{}"'

# 4. evaluation (thresholds tuned on validation, metrics on test)
for r in "${RUNS[@]}"; do set -- $r; t=$(tagof "$@")
  [[ -f $OUT/models/$t.pt && ! -f $OUT/eval/$t.json ]] && { log "eval $t"; python scripts/evaluate.py --ckpt "$OUT/models/$t.pt"; }
done

# 4b. validation predictions of the 4 s models (alert levels fixed on NZ validation data)
for r in "${RUNS[@]:0:14}"; do set -- $r; t=$(tagof "$@")
  [[ -f $OUT/eval/$t.json && ! -f $OUT/eval/${t}_validation_predictions.csv.gz ]] && \
    python scripts/dump_predictions.py --ckpt "$OUT/models/$t.pt" --split validation --out "$OUT"
done

# 5. Christchurch case study (4 s models)
if [[ -f data/christchurch/christchurch.hdf5 ]]; then
  for r in "${RUNS[@]:0:14}"; do set -- $r; t=$(tagof "$@")
    [[ -f $OUT/christchurch/${t}_summary.json ]] || python scripts/christchurch_eval.py --ckpt "$OUT/models/$t.pt" --eval-json "$OUT/eval/$t.json"
  done
  python paper/christchurch_rules.py "$OUT"
  # 5b. Pd on all NZ records, threshold fitted on the training split (CPU, about an hour)
  [[ -f $OUT/pd/pd_calibration.json ]] || python scripts/pd_calibration.py --out "$OUT"
  python paper/christchurch_calibrated.py "$OUT"
else log "data/christchurch/christchurch.hdf5 not found: skipping the case study"; fi

# 6. complexity and latency
log "benchmark"; python scripts/bench_compute.py || log "benchmark FAILED"
python paper/main_tables.py "$OUT" || true
python paper/fig_paper.py "$OUT" || true
log "ALL DONE"
