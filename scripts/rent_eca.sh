#!/usr/bin/env bash
# One rental on a 5090: the cellular automaton runs, RoPE against learned positions.
#
# On the rented machine (a PyTorch image), from the repo root after cloning:
#   nohup scripts/rent_eca.sh > logs/rent.log 2>&1 < /dev/null &
#   tail -f logs/rent/status.txt            # one line per step: ok, FAILED or skipped
#   cat logs/rent/summary.txt               # the last scores of every run, once they are done
# Then fetch the results to the Mac:   rsync -av HOST:noema/{runs,checkpoints,logs} .
#
# Nothing to upload: the data is made as the model trains.
# The smoke step comes first. Read its speed in logs/rent/rent_smoke.log (the first line is slow, it
# compiles) before you let the rest run: "step 100 ... (Ns)" minus "step 50 ... (Ns)" is 50 steps.
# Each step logs to logs/rent/rent_<step>.log. A step whose output exists is skipped, so after a
# problem, run the script again. A failed step doesn't stop the others.
#
# Settings, as environment variables (the smoke test on the Mac makes them tiny):
STEPS=${STEPS:-3000}                    # training steps in each run
SMOKE_STEPS=${SMOKE_STEPS:-100}
SEEDS=${SEEDS:-"0 1"}
VARIANTS=${VARIANTS:-"rope positions"}  # rope: RoPE. positions: a learned table in its place
PIP=${PIP:-1}                           # 0: the packages are there already (the Mac)
EXTRA=${EXTRA:-}                        # more arguments for train_eca, e.g. "--radius 2" or "--emb 32 --heads 2 --layers 2"

set -u
cd "$(dirname "$0")/.."
LOG=logs/rent
STATUS=$LOG/status.txt
SUMMARY=$LOG/summary.txt
mkdir -p "$LOG" runs checkpoints
CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$CUDA" = 1 ]; then COMPILE=--compile; else COMPILE=; fi

step() {    # step NAME OUTPUT COMMAND...: run COMMAND, unless OUTPUT exists ("" = always run)
    local name=$1 out=$2; shift 2
    if [ -n "$out" ] && [ -e "$out" ]; then
        echo "$(date +%H:%M:%S) $name: skipped, $out exists" | tee -a "$STATUS"; return
    fi
    echo "$(date +%H:%M:%S) $name: started" | tee -a "$STATUS"
    if "$@" > "$LOG/rent_$name.log" 2>&1; then
        echo "$(date +%H:%M:%S) $name: ok" | tee -a "$STATUS"
    else
        echo "$(date +%H:%M:%S) $name: FAILED, see $LOG/rent_$name.log" | tee -a "$STATUS"
    fi
}

flags_for() {   # the train_eca flags of a variant
    case $1 in
        rope) echo "" ;;
        positions) echo "--learned-positions" ;;
        *) echo "unknown variant $1" >&2; return 1 ;;
    esac
}

# ---------- setup and checks ----------
if [ "$PIP" = 1 ]; then step pip "" pip install -q -r requirements.txt; fi
step tests "" python -m pytest -q
step smoke "" python -m noema.train_eca --steps "$SMOKE_STEPS" --log-every $((SMOKE_STEPS / 2)) --tag smoke $COMPILE $EXTRA

# ---------- the runs: seed 0 for every variant first, so a partial result is still a fair comparison ----------
for seed in $SEEDS; do
    for variant in $VARIANTS; do
        tag=${variant}_s$seed
        step "run_$tag" "checkpoints/eca_$tag.pt" \
            python -m noema.train_eca --steps "$STEPS" --seed "$seed" --tag "$tag" $(flags_for "$variant") $COMPILE $EXTRA
    done
done

# ---------- the scores: the last evaluation and the accuracy by row of each run ----------
: > "$SUMMARY"
for seed in $SEEDS; do
    for variant in $VARIANTS; do
        log="$LOG/rent_run_${variant}_s$seed.log"
        [ -f "$log" ] || continue
        { echo "== $variant, seed $seed"; grep "^step" "$log" | tail -1; grep "^accuracy by row" "$log"; echo; } >> "$SUMMARY"
    done
done
echo "$(date +%H:%M:%S) all done: see $SUMMARY" | tee -a "$STATUS"
