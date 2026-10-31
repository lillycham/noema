#!/usr/bin/env bash
# The rung 5 batch, for shukka: free strides, and more seeds of the 2D world with checkpoints along the curve.
#
#   shukka run --job ./noema#smoke             a few cents on a small card: does it all run on this host?
#   shukka run --job ./noema#rung5             the batch
#
# The flake gives PYTHONPATH and python. Everything is written under $SHUKKA_OUT, so it comes home with
# the logs when the run ends, also when it fails. A failed run doesn't stop the others.
#
# Settings, as arguments: smoke | batch.  JOBS is how many runs share the GPU.
set -u
MODE=${1:-smoke}
JOBS=${JOBS:-6}
COMPILE=${COMPILE:---compile}
cd "${SHUKKA_OUT:-.}"
mkdir -p logs checkpoints runs
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count() and torch.cuda.get_device_name(0))" | tee logs/gpu.txt

padded() { echo "padded_$1 python -m noema.train_padded --steps $2 --seed $1 --tag s$1 $COMPILE"; }
world2d() {  # world2d NAME SEED STEPS FLAGS
    echo "2d_$1$2 python -m noema.train_2d --steps $3 --seed $2 --tag $1$2 --log-every 50 --save-at 500 1000 1500 2000 3000 4000 $COMPILE ${4:-}"
}

case $MODE in
    smoke)
        COMPILE= padded 0 200 > plan.txt
        padded 1 200 >> plan.txt
        COMPILE= world2d smoke 0 200 >> plan.txt
        world2d smokec 0 200 >> plan.txt
        JOBS=1
        ;;
    batch)
        for s in 0 1 2 3; do padded $s 32000; done > plan.txt
        for s in 2 3 4 5 6 7 8 9; do world2d pos $s 6000 --learned-positions; done >> plan.txt
        for s in 2 3 4 5; do world2d rope $s 6000; done >> plan.txt
        ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

cat plan.txt
run_one() {
    name=${1%% *}; command=${1#* }
    echo "$(date +%H:%M:%S) $name started"
    if $command > "logs/$name.log" 2>&1; then echo "$(date +%H:%M:%S) $name ok"; else echo "$(date +%H:%M:%S) $name FAILED"; tail -n 5 "logs/$name.log"; fi
    tail -n 1 "logs/$name.log"
}
export -f run_one
tr '\n' '\0' < plan.txt | xargs -0 -P "$JOBS" -I{} bash -c 'run_one "$1"' _ {}
echo all done
