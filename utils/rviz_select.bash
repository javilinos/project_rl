#!/bin/bash
# Interactive RViz selector for per-drone DDS-isolated training.
#
# Each drone runs on `base_domain_id + index` (see launch_as2.bash -d).
# This script displays a menu of drones, launches RViz on the chosen
# drone's ROS_DOMAIN_ID, and re-launches when the selection changes.
# Only ONE RViz process runs at a time, so it scales to as many drones
# as you want without memory pressure.
#
# Usage:
#   utils/rviz_select.bash <base_domain_id> <namespaces_csv>
#
# Example (10 drones starting at domain 42):
#   utils/rviz_select.bash 42 drone0,drone1,drone2,drone3,drone4,drone5,drone6,drone7,drone8,drone9

set -u

base_domain="${1:-}"
namespaces_csv="${2:-}"

if [ -z "$base_domain" ] || [ -z "$namespaces_csv" ]; then
    echo "usage: $0 <base_domain_id> <namespaces_csv>" >&2
    exit 1
fi

IFS=',' read -r -a namespaces <<< "$namespaces_csv"
n=${#namespaces[@]}

rviz_pid=""

kill_current_rviz() {
    if [ -n "$rviz_pid" ] && kill -0 "$rviz_pid" 2>/dev/null; then
        # Send SIGTERM to the ros2 launch process group; it will reap
        # rviz + helper nodes. Wait a bit so DDS releases its handles
        # before we open a new domain.
        kill -- -"$rviz_pid" 2>/dev/null || kill "$rviz_pid" 2>/dev/null
        wait "$rviz_pid" 2>/dev/null || true
    fi
    rviz_pid=""
}

cleanup() {
    kill_current_rviz
}
trap cleanup EXIT INT TERM

launch_rviz_for() {
    local idx="$1"
    local drone="${namespaces[$idx]}"
    local domain=$((base_domain + idx))

    kill_current_rviz

    echo
    echo "==> launching RViz for $drone on ROS_DOMAIN_ID=$domain"
    # `setsid` so the background process gets its own process group; that
    # lets `kill -- -PID` cleanly take down the whole launch tree later.
    setsid env ROS_DOMAIN_ID="$domain" ros2 launch as2_visualization swarm_viz.launch.py \
        namespace_list:="$drone" \
        rviz_config:=config_ground_station/rviz2_config.rviz \
        drone_model:=quadrotor_base \
        >/tmp/rviz_select_${drone}.log 2>&1 &
    rviz_pid=$!
}

print_menu() {
    echo
    echo "============================================================"
    echo "  RViz drone selector  (base_domain_id=$base_domain)"
    echo "============================================================"
    for i in "${!namespaces[@]}"; do
        printf "    %2d) %-12s  domain=%d\n" "$i" "${namespaces[$i]}" "$((base_domain + i))"
    done
    echo "     q) quit"
    echo "------------------------------------------------------------"
}

# Auto-launch drone 0 so the user has *something* on screen immediately.
launch_rviz_for 0

while true; do
    print_menu
    read -r -p "Select drone (0-$((n - 1))): " choice
    if [ "$choice" = "q" ] || [ "$choice" = "Q" ]; then
        break
    fi
    if ! [[ "$choice" =~ ^[0-9]+$ ]]; then
        echo "  ! not a number"
        continue
    fi
    if [ "$choice" -ge "$n" ]; then
        echo "  ! out of range (max $((n - 1)))"
        continue
    fi
    launch_rviz_for "$choice"
done
