#!/bin/bash
# Record one AnyDexGrasp work loop: episode.sh POOL SEED ATTEMPTS OUT_DIR
# Restarts the sim (sort2, headless) with the pool, needs adg_server.py running; writes OUT_DIR/pool<P>.mp4.
set -u
POOL=$1 SEED=$2 N=$3 OUT=$4
W=$(cd "$(dirname "$0")" && pwd)
POLY=/home/litt/Projects/nssr-polygon
mkdir -p "$OUT" /tmp/adg
SIM_LOG=$OUT/sim_pool$POOL.log
(cd $POLY && HACK_RENDER_MODE=${HACK_RENDER_MODE:-quality} script -qfec "bash hackathon/tools/task.sh sort2 --pool $POOL --seed $SEED --headless" /dev/null \
    >"$SIM_LOG" 2>&1 </dev/null) &
SIM=$!
until grep -q "cam_head=[1-9]" "$SIM_LOG" 2>/dev/null; do
    kill -0 $SIM 2>/dev/null || {
        echo "sim died"
        exit 1
    }
    sleep 2
done
sleep 5
echo "sim up: pool $POOL seed $SEED"
cd "$W" || echo "Failed cd"
rm -f client.log /tmp/adg/cmd.fifo
./ros.sh python3 -u adg_client.py >client.log 2>&1 &
CLIENT=$!
until grep -q "ready for commands" client.log; do
    kill -0 $CLIENT || {
        echo "client died"
        exit 1
    }
    sleep 1
done
./ros.sh python3 -u record.py "$OUT/pool$POOL.raw.mp4" "AnyDexGrasp | sort2 pool $POOL seed $SEED" "$W/client.log" 270 \
    >"$OUT/record_pool$POOL.log" 2>&1 &
sleep 5
CMD_TIMEOUT=$((N * 400)) ./cmd.sh auto "$N" | grep -E "^auto|holding|empty|grasp:|^tray|^skip|^targets" | tee "$OUT/auto_pool$POOL.txt"
sleep 3
./ros.sh pkill -INT -f "[r]ecord.py"
sleep 3
echo quit >/tmp/adg/cmd.fifo
wait $CLIENT
ffmpeg -loglevel error -y -i "$OUT/pool$POOL.raw.mp4" -c:v libx264 -crf 22 -pix_fmt yuv420p "$OUT/pool$POOL.mp4" &&
    rm "$OUT/pool$POOL.raw.mp4"
cp "$W/client.log" "$OUT/client_pool$POOL.log"
echo "video: $OUT/pool$POOL.mp4"
