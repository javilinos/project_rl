#!/bin/bash

usage() {
    echo "  options:"
    echo "      -c: motion controller plugin (pid_speed_controller, differential_flatness_controller), choices: [pid, df]. Default: pid"
    echo "      -m: multi agent. Default not set"
    echo "      -n: select drones namespace to launch, values are comma separated. By default, it will get all drones from world description file"
    echo "      -g: launch using gnome-terminal instead of tmux. Default not set"
    echo "      -d: base ROS_DOMAIN_ID. Each drone gets base + index; e.g. -d 42 gives drone0→42, drone1→43, … Default: unset (all share the inherited domain)."
}

# Initialize variables with default values
motion_controller_plugin="pid"
swarm="false"
drones_namespace_comma=""
use_gnome="false"
# Empty = leave ROS_DOMAIN_ID alone (current behaviour). Set with -d to enable
# per-drone DDS isolation: each drone runs on base_domain_id + drone_index.
base_domain_id=""

# Arg parser
while getopts "cmn:gd:" opt; do
  case ${opt} in
    c )
      motion_controller_plugin="${OPTARG}"
      ;;
    m )
      swarm="true"
      ;;
    n )
      drones_namespace_comma="${OPTARG}"
      ;;
    g )
      use_gnome="true"
      ;;
    d )
      base_domain_id="${OPTARG}"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[wrt]$ ]]; then
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
IFS=',' read -r -a drone_namespaces <<< "$drones_namespace_comma"

# Check if motion controller plugins are valid
case ${motion_controller_plugin} in
  pid )
    motion_controller_plugin="pid_speed_controller"
    ;;
  df )
    motion_controller_plugin="differential_flatness_controller"
    ;;
  * )
    echo "Invalid motion controller plugin: ${motion_controller_plugin}" >&2
    usage
    exit 1
    ;;
esac

# Select between tmux and gnome-terminal
tmuxinator_mode="start"
tmuxinator_end="wait"
tmp_file="/tmp/as2_project_launch_${drone_namespaces[@]}.txt"
if [[ ${use_gnome} == "true" ]]; then
  tmuxinator_mode="debug"
  tmuxinator_end="> ${tmp_file} && python3 utils/tmuxinator_to_genome.py -p ${tmp_file} && wait"
fi

# Launch aerostack2 for each drone namespace
drone_index=0
for namespace in ${drone_namespaces[@]}; do
  base_launch="false"
  if [[ ${namespace} == ${drone_namespaces[0]} ]]; then
    base_launch="true"
  fi
  # Per-drone ROS_DOMAIN_ID: empty when -d wasn't passed so the tmuxinator
  # template can leave the inherited domain alone. Otherwise base+index.
  if [[ -n "${base_domain_id}" ]]; then
    drone_domain_id=$((base_domain_id + drone_index))
  else
    drone_domain_id=""
  fi
  eval "tmuxinator ${tmuxinator_mode} -n ${namespace} -p tmuxinator/aerostack2.yaml \
    drone_namespace=${namespace} \
    simulation_config_file=${simulation_config} \
    motion_controller_plugin=${motion_controller_plugin} \
    base_launch=${base_launch} \
    ros_domain_id=${drone_domain_id} \
    ${tmuxinator_end}"

  sleep 0.1 # Wait for tmuxinator to finish
  drone_index=$((drone_index + 1))
done

# Attach to tmux session
if [[ ${use_gnome} == "false" ]]; then
  tmux attach-session -t ${drone_namespaces[0]}
# If tmp_file exists, remove it
elif [[ -f ${tmp_file} ]]; then
  rm ${tmp_file}
fi
