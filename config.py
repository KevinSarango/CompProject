"""
config.py

Central project configuration.

This file is the single source of truth for:
- load-balancing limit used by the RL training environment
- load-balancing limit used by the deployed Ryu RL controller
- diamond topology link parameters
- pingall retry behavior
"""

# One shared load-balancing limit used across training and controller logic.
LOAD_BALANCE_LIMIT_KB = 3000.0

# Shared load decay used by the training environment and RL controller.
LOAD_DECAY_FACTOR = 0.85

# Approximate flow size used by the deployed Ryu RL controller's load estimator.
DEFAULT_FLOW_SIZE_KB = 500.0

# Symmetric diamond topology link settings.
LINK_BW_MBPS = 10
LINK_DELAY = "5ms"

# Connectivity fallback settings.
PINGALL_ATTEMPTS = 3
PINGALL_RETRY_WAIT_SECONDS = 2
