"""
Shared configuration for the SDN RL project.

The goal is to keep important constants in one place so the training
simulation and Ryu deployment use the same assumptions.
"""

# Shared load-balancing capacity limit used by both paths.
LOAD_BALANCE_LIMIT_KB = 2750.0

# Training/deployment load model.
LOAD_DECAY_FACTOR = 0.85
DEFAULT_FLOW_SIZE_KB = 500.0

# Symmetric Mininet link parameters.
LINK_BW_MBPS = 10
LINK_DELAY = "5ms"

# Pingall fallback settings.
PINGALL_ATTEMPTS = 3
PINGALL_RETRY_WAIT_SECONDS = 2

# State bin thresholds.
# Utilization difference is computed as upper_util - lower_util.
# If upper_util is lower by more than 0.15, upper is less utilized.
# If lower_util is lower by more than 0.15, lower is less utilized.
UTILIZATION_DIFF_THRESHOLD = 0.15

# Flow size bins for incoming demand.
SMALL_FLOW_KB = 350.0
MEDIUM_FLOW_KB = 600.0

# Reward weights.
GAMMA_PACKET_LOSS = 2.0
GAMMA_DELAY = 1.5
GAMMA_THROUGHPUT = 1.0
GAMMA_ACTION_IMPACT = 1.0
GAMMA_IMBALANCE = 0.5
GAMMA_SWITCHING = 0.1
BASE_REWARD = 1.0
