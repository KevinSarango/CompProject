"""
Shared configuration for the SDN RL project.

TOPO_MODE controls which experiment is active:
    diamond     = original 2-path topology
    three_path  = new 3-path asymmetric topology

Use:
    TOPO_MODE=diamond python3 train_rl_agent.py
    TOPO_MODE=three_path python3 train_rl_agent.py
"""

import os


TOPO_MODE = os.environ.get("TOPO_MODE", "diamond").strip().lower()

if TOPO_MODE not in ["diamond", "three_path"]:
    raise ValueError(
        f"Invalid TOPO_MODE={TOPO_MODE}. Use 'diamond' or 'three_path'."
    )


# Shared demand / load settings.
LOAD_BALANCE_LIMIT_KB = 2750.0
LOAD_DECAY_FACTOR = 0.85
DEFAULT_FLOW_SIZE_KB = 500.0

# Pingall fallback settings.
PINGALL_ATTEMPTS = 3
PINGALL_RETRY_WAIT_SECONDS = 2

# Flow size bins.
SMALL_FLOW_KB = 350.0
MEDIUM_FLOW_KB = 600.0

# Utilization bin threshold.
UTILIZATION_DIFF_THRESHOLD = 0.15

# Reward weights.
GAMMA_PACKET_LOSS = 2.0
GAMMA_DELAY = 1.5
GAMMA_THROUGHPUT = 1.0
GAMMA_ACTION_IMPACT = 1.0
GAMMA_IMBALANCE = 0.5
GAMMA_SWITCHING = 0.1
BASE_REWARD = 1.0


# Original diamond topology link parameters.
DIAMOND_LINK_BW_MBPS = 10
DIAMOND_LINK_DELAY = "5ms"


# Three-path Mininet link parameters.
# Path 0: low delay, low bandwidth
# Path 1: balanced
# Path 2: high bandwidth, high delay
THREE_PATH_LINKS = {
    "low_delay": {
        "bw": 5,
        "delay": "2ms",
    },
    "balanced": {
        "bw": 10,
        "delay": "5ms",
    },
    "high_bw": {
        "bw": 15,
        "delay": "12ms",
    },
}


PATH_CONFIGS = {
    "diamond": {
        "paths": ["upper", "lower"],
        "q_table_file": "data/q_table_diamond.json",
        "training_rewards_file": "data/training_rewards_diamond.csv",
        "training_steps_file": "data/training_steps_diamond.csv",
        "fifo_metrics_file": "data/diamond_fifo_metrics.csv",
        "rl_metrics_file": "data/diamond_rl_metrics.csv",
        "fifo_traffic_file": "data/diamond_fifo_traffic_metrics.csv",
        "rl_traffic_file": "data/diamond_rl_traffic_metrics.csv",
        "plot_prefix": "diamond",

        # Diamond paths are symmetric.
        "path_capacity_kb": {
            "upper": 2750.0,
            "lower": 2750.0,
        },
        "path_delay_factor": {
            "upper": 1.0,
            "lower": 1.0,
        },
    },

    "three_path": {
        "paths": ["low_delay", "balanced", "high_bw"],
        "q_table_file": "data/q_table_three_path.json",
        "training_rewards_file": "data/training_rewards_three_path.csv",
        "training_steps_file": "data/training_steps_three_path.csv",
        "fifo_metrics_file": "data/three_path_fifo_metrics.csv",
        "rl_metrics_file": "data/three_path_rl_metrics.csv",
        "fifo_traffic_file": "data/three_path_fifo_traffic_metrics.csv",
        "rl_traffic_file": "data/three_path_rl_traffic_metrics.csv",
        "plot_prefix": "three_path",

        # Training/deployment model for the 3-path topology.
        #
        # low_delay:
        #   lower capacity, lower delay
        #
        # balanced:
        #   medium capacity, medium delay
        #
        # high_bw:
        #   higher capacity, higher delay
        "path_capacity_kb": {
            "low_delay": 1500.0,
            "balanced": 2750.0,
            "high_bw": 4000.0,
        },
        "path_delay_factor": {
            "low_delay": 0.70,
            "balanced": 1.00,
            "high_bw": 1.50,
        },
    },
}


ACTIVE_CONFIG = PATH_CONFIGS[TOPO_MODE]
PATHS = ACTIVE_CONFIG["paths"]
NUM_PATHS = len(PATHS)

ACTION_TO_PATH = {
    str(index): path_name
    for index, path_name in enumerate(PATHS)
}

PATH_TO_ACTION = {
    path_name: str(index)
    for index, path_name in enumerate(PATHS)
}

PATH_CAPACITY_KB = [
    ACTIVE_CONFIG["path_capacity_kb"][path_name]
    for path_name in PATHS
]

PATH_DELAY_FACTOR = [
    ACTIVE_CONFIG["path_delay_factor"][path_name]
    for path_name in PATHS
]


Q_TABLE_FILE = ACTIVE_CONFIG["q_table_file"]
TRAINING_REWARDS_FILE = ACTIVE_CONFIG["training_rewards_file"]
TRAINING_STEPS_FILE = ACTIVE_CONFIG["training_steps_file"]

FIFO_METRICS_FILE = ACTIVE_CONFIG["fifo_metrics_file"]
RL_METRICS_FILE = ACTIVE_CONFIG["rl_metrics_file"]

FIFO_TRAFFIC_FILE = ACTIVE_CONFIG["fifo_traffic_file"]
RL_TRAFFIC_FILE = ACTIVE_CONFIG["rl_traffic_file"]

PLOT_PREFIX = ACTIVE_CONFIG["plot_prefix"]
PLOTS_DIR = "data/plots"
