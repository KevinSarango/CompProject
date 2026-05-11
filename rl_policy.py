import json
import os

import numpy as np

from config import (
    ACTION_TO_PATH,
    DEMAND_BINS,
    MAX_DELAY_SCORE,
    MAX_FLOW_SIZE_KB,
    NUM_PATHS,
    PATH_CAPACITY_KB,
    PATH_DELAY_FACTOR,
    Q_TABLE_FILE,
    STATE_BINS,
)


CURRENT_STATE = None


PATH_CAPACITY_ARRAY = np.array(PATH_CAPACITY_KB, dtype=float)
PATH_DELAY_FACTOR_ARRAY = np.array(PATH_DELAY_FACTOR, dtype=float)


def load_q_table():
    if not os.path.exists(Q_TABLE_FILE):
        print(f"[WARN] {Q_TABLE_FILE} not found. Using fallback policy.")
        return None

    with open(Q_TABLE_FILE, "r") as f:
        return json.load(f)


Q_TABLE = load_q_table()


def discretize(value, min_value, max_value, bins):
    if bins <= 1 or max_value <= min_value:
        return 0

    value = max(min_value, min(float(value), max_value))
    scaled = (value - min_value) / (max_value - min_value)
    index = int(scaled * bins)
    return min(index, bins - 1)


def get_path_utilizations(path_loads):
    loads = np.array(path_loads, dtype=float)
    return loads / PATH_CAPACITY_ARRAY


def get_path_delay_scores(path_loads):
    utilizations = get_path_utilizations(path_loads)
    raw_delay_scores = utilizations * 100.0 * PATH_DELAY_FACTOR_ARRAY
    return np.clip(raw_delay_scores / MAX_DELAY_SCORE, 0.0, 1.0)


def build_state(path_loads, flow_size_kb, previous_action):
    """
    Build the same multipath ratio-state key used during training:
        least_utilized_path_utilization_spread_bin_delay_spread_bin_demand_bin_previous_action
    """
    utilizations = get_path_utilizations(path_loads)
    delay_scores = get_path_delay_scores(path_loads)

    least_path = int(np.argmin(utilizations))
    util_spread = float(np.max(utilizations) - np.min(utilizations))
    delay_spread = float(np.max(delay_scores) - np.min(delay_scores))

    util_spread_bin = discretize(util_spread, 0.0, 1.0, STATE_BINS)
    delay_spread_bin = discretize(delay_spread, 0.0, 1.0, STATE_BINS)
    demand_bin = discretize(flow_size_kb, 0.0, MAX_FLOW_SIZE_KB, DEMAND_BINS)

    return (
        f"{least_path}_"
        f"{util_spread_bin}_"
        f"{delay_spread_bin}_"
        f"{demand_bin}_"
        f"{previous_action}"
    )


def set_network_state(state):
    global CURRENT_STATE
    CURRENT_STATE = str(state)


def choose_path(src_ip, dst_ip, state=None):
    selected_state = str(state) if state is not None else CURRENT_STATE

    if Q_TABLE is not None and selected_state in Q_TABLE:
        action = max(Q_TABLE[selected_state], key=Q_TABLE[selected_state].get)
        return ACTION_TO_PATH[action]

    # Fallback if Q-table is missing or state is unknown.
    dst_num = int(dst_ip.split(".")[-1])
    fallback_action = dst_num % NUM_PATHS
    return ACTION_TO_PATH[str(fallback_action)]
