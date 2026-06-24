#!/usr/bin/env bash
# Live monitor for train_eval_svc.sh
# Run in a second terminal: bash monitor_svc.sh
# Refreshes every 3 seconds. Ctrl-C to quit.

LOGDIR=logs/svc
POLICY_LOG="$LOGDIR/policy_ids.txt"
TOTAL=8

TAGS=(
    sarsa_rand_grid sarsa_ames
    a2c_grid_road   a2c_rand_grid   a2c_ames
    ppo_grid_road   ppo_rand_grid   ppo_ames
)

# ── helpers ───────────────────────────────────────────────────────────────────

progress_bar() {
    local cur=$1 total=$2 width=${3:-40}
    local pct=$(( cur * 100 / (total > 0 ? total : 1) ))
    local filled=$(( cur * width / (total > 0 ? total : 1) ))
    local bar=""
    for ((i=0; i<filled; i++));  do bar+="█"; done
    for ((i=filled; i<width; i++)); do bar+="░"; done
    printf "[%s] %3d%%" "$bar" "$pct"
}

# Extract current step and total from the active log.
# Sets globals: STEP, TOTAL_STEPS, LAST_REWARD, LAST_EPSILON, LAST_LOSS
parse_log() {
    local log=$1 tag=$2
    STEP=0; TOTAL_STEPS=1; LAST_REWARD=""; LAST_EPSILON=""; LAST_LOSS=""

    if [[ "$tag" == sarsa_* ]]; then
        TOTAL_STEPS=5000
        # "Episode 42/5000  Total Reward: -12.34  Epsilon: 0.7812"
        local line
        line=$(grep "^Episode " "$log" 2>/dev/null | tail -1)
        if [ -n "$line" ]; then
            STEP=$(echo "$line" | awk '{print $2}' | cut -d/ -f1)
            LAST_REWARD=$(echo "$line" | awk '{print $5}')
            LAST_EPSILON=$(echo "$line" | awk '{print $7}')
        fi
    else
        TOTAL_STEPS=100
        # A2C: "Iter 7/100  pg_loss=0.0321  v_loss=0.1234  mean_ep_ret=-45.23"
        # PPO: "Iter 7/100  Episode 3  Reward: -12.34"
        local line
        line=$(grep "^Iter " "$log" 2>/dev/null | tail -1)
        if [ -n "$line" ]; then
            STEP=$(echo "$line" | awk '{print $2}' | cut -d/ -f1)
            LAST_LOSS=$(echo "$line" | grep -oP 'pg_loss=\S+' | head -1)
            LAST_REWARD=$(echo "$line" | grep -oP '(?<=mean_ep_ret=)\S+' | head -1)
            if [ -z "$LAST_REWARD" ]; then
                # PPO format: "Iter N/M  Episode K  Reward: X"
                LAST_REWARD=$(echo "$line" | grep -oP '(?<=Reward: )\S+' | head -1)
            fi
        fi
    fi
    STEP=${STEP:-0}
}

clear_screen() { clear; }

# ── main loop ─────────────────────────────────────────────────────────────────

while true; do
    clear_screen

    DONE=0
    if [ -f "$POLICY_LOG" ]; then
        DONE=$(grep -c '=' "$POLICY_LOG" 2>/dev/null; true)
        DONE=$(( DONE + 0 ))   # coerce to int; handles empty string
    fi

    echo "══════════════════════════════════════════════════════════════"
    printf "  SVC Batch  %d / %d   %s\n" "$DONE" "$TOTAL" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "  Overall:   "
    progress_bar "$DONE" "$TOTAL" 44
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo ""

    # ── per-run status ────────────────────────────────────────────────────────
    for tag in "${TAGS[@]}"; do
        log="$LOGDIR/train_${tag}.log"
        pid_entry=$(grep "^${tag}=" "$POLICY_LOG" 2>/dev/null | head -1)

        if [ -n "$pid_entry" ]; then
            pid=${pid_entry#*=}
            # Grab final reward from completed log if available
            if [[ "$tag" == sarsa_* ]]; then
                last=$(grep "^Episode " "$log" 2>/dev/null | tail -1 | awk '{print "ep="$2" r="$5}')
            else
                last_line=$(grep "^Iter " "$log" 2>/dev/null | tail -1)
                last=$(echo "$last_line" | grep -oP 'mean_ep_ret=\S+' | head -1)
                if [ -z "$last" ]; then
                    rew=$(echo "$last_line" | grep -oP '(?<=Reward: )\S+' | head -1)
                    [ -n "$rew" ] && last="ret=${rew}"
                fi
            fi
            printf "  ✓  %-22s  %s  %s\n" "$tag" "$pid" "$last"

        elif [ -f "$log" ]; then
            parse_log "$log" "$tag"
            printf "  ▶  %-22s  " "$tag"
            progress_bar "$STEP" "$TOTAL_STEPS" 30
            if [ -n "$LAST_REWARD" ]; then
                printf "  ret=%-8s" "$LAST_REWARD"
            fi
            if [ -n "$LAST_EPSILON" ]; then
                printf "  ε=%s" "$LAST_EPSILON"
            fi
            if [ -n "$LAST_LOSS" ]; then
                printf "  %s" "$LAST_LOSS"
            fi
            echo ""
        else
            printf "  ○  %s\n" "$tag"
        fi
    done

    echo ""

    # ── live tail of the active log ───────────────────────────────────────────
    ACTIVE_LOG=$(ls -1t "$LOGDIR"/train_*.log 2>/dev/null | head -1)
    if [ -n "$ACTIVE_LOG" ]; then
        ACTIVE_TAG=$(basename "$ACTIVE_LOG" .log | sed 's/^train_//')
        echo "── $ACTIVE_TAG ─────────────────────────────────────────────────"
        grep -E "^(Episode|Iter|Checkpoint|Policy saved)" "$ACTIVE_LOG" 2>/dev/null | tail -6
    fi

    # ── latest eval ───────────────────────────────────────────────────────────
    EVAL_LOG=$(ls -1t "$LOGDIR"/eval_*.log 2>/dev/null | head -1)
    if [ -n "$EVAL_LOG" ]; then
        echo ""
        echo "── eval: $(basename "$EVAL_LOG" .log | sed 's/^eval_//') ──────────────────────────────────────"
        tail -6 "$EVAL_LOG" 2>/dev/null
    fi

    sleep 3
done
