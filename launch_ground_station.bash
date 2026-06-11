#!/bin/bash

usage() {
    echo "  options:"
    echo "      -m: multi agent. Default not set"
    echo "      -t: launch keyboard teleoperation. Default not launch"
    echo "      -v: open rviz. Default launch"
    echo "      -r: record rosbag. Default not launch"
    echo "      -n: drone namespaces, comma separated. Default get from world description config file"
    echo "      -g: launch using gnome-terminal instead of tmux. Default not set"
    echo "      -d: ROS_DOMAIN_ID to observe. RViz and the alphanumeric viewer will only see drones on this domain. Use to focus on a single drone trained with per-drone DDS isolation."
    echo "      -s: launch the per-drone RViz selector (utils/rviz_select.bash). -d sets the base domain; one drone is shown at a time, swap with the menu."
}

# Initialize variables with default values
swarm="false"
keyboard_teleop="false"
rviz="true"
rosbag="false"
drones_namespace_comma=""
use_gnome="false"
# Empty = inherit the parent shell's ROS_DOMAIN_ID. Set to a specific value to
# pin this ground station (rviz + viewer) to one drone's DDS domain.
ros_domain_id=""
# When set, the rviz pane runs the interactive selector instead of a single
# fixed RViz instance. Pair with -d <base> and -n / -m so the selector knows
# what to switch between.
selector_mode="false"

# Parse command line arguments
while getopts "mtvrn:gd:s" opt; do
  case ${opt} in
    m )
      swarm="true"
      ;;
    t )
      keyboard_teleop="true"
      ;;
    v )
      rviz="false"
      ;;
    r )
      rosbag="true"
      ;;
    n )
      drones_namespace_comma="${OPTARG}"
      ;;
    g )
      use_gnome="true"
      ;;
    d )
      ros_domain_id="${OPTARG}"
      ;;
    s )
      selector_mode="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[swrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

# Set simulation world description config file
if [[ ${swarm} == "true" ]]; then
  simulation_config="config/world_swarm.yaml"
else
  simulation_config="config/world.yaml"
fi

# If no drone namespaces are provided, get them from the world description config file
if [ -z "$drones_namespace_comma" ]; then
  drones_namespace_comma=$(python3 utils/get_drones.py -p ${simulation_config} --sep ',')
fi

# Select between tmux and gnome-terminal
tmuxinator_mode="start"
tmuxinator_end="wait"
tmp_file="/tmp/as2_project_launch_${drone_namespaces[@]}.txt"
if [[ ${use_gnome} == "true" ]]; then
  tmuxinator_mode="debug"
  tmuxinator_end="> ${tmp_file} && python3 utils/tmuxinator_to_genome.py -p ${tmp_file} && wait"
fi

# Launch aerostack2 ground station
eval "tmuxinator ${tmuxinator_mode} -n ground_station -p tmuxinator/ground_station.yaml \
  drone_namespace=${drones_namespace_comma} \
  keyboard_teleop=${keyboard_teleop} \
  rviz=${rviz} \
  rosbag=${rosbag} \
  ros_domain_id=${ros_domain_id} \
  selector_mode=${selector_mode} \
  ${tmuxinator_end}"

# Attach to tmux session
if [[ ${use_gnome} == "false" ]]; then
  tmux attach-session -t ground_station
# If tmp_file exists, remove it
elif [[ -f ${tmp_file} ]]; then
  rm ${tmp_file}
fi