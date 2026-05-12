import json
import os

from config import (
    ACTION_TO_PATH,
    DEFAULT_FLOW_SIZE_KB,
    MEDIUM_FLOW_KB,
    NUM_PATHS,
    PATH_CAPACITY_KB,
    Q_TABLE_FILE,
    SMALL_FLOW_KB,
    UTILIZATION_DIFF_THRESHOLD,
)


CURRENT_STATE = None


def load_q_table():
    if not os.path.exists(Q_TABLE_FILE):
        print(f"[WARN] {Q_TABLE_FILE} not found. Using fallback policy.")
        return None

    with open(Q_TABLE_FILE, "r") as f:
        return json.load(f)


Q_TABLE = load_q_table()


def get_path_utilizations(path_loads):
    utilizations = []

    for index, load in enumerate(path_loads):
        capacity = PATH_CAPACITY_KB[index]
        utilizations.append(load / capacity)

    return utilizations


def get_least_utilized_path_bin(path_loads):
    """
    Returns the path index with the lowest utilization.

    Utilization is path_load / path_specific_capacity.
    This matters because the three-path topology has unequal capacities.
    """

    utilizations = get_path_utilizations(path_loads)

    min_util = min(utilizations)
    max_util = max(utilizations)

    # If all paths are close, treat it as the middle/balanced state
    # when three paths are available.
    if max_util - min_util <= UTILIZATION_DIFF_THRESHOLD:
        if NUM_PATHS == 3:
            return 1
        return 0

    return utilizations.index(min_util)


def get_demand_bin(flow_size_kb):
    if flow_size_kb <= SMALL_FLOW_KB:
        return 0

    if flow_size_kb <= MEDIUM_FLOW_KB:
        return 1

    return 2


def build_state(path_loads, flow_size_kb, previous_action):
    least_utilized_bin = get_least_utilized_path_bin(path_loads)
    demand_bin = get_demand_bin(flow_size_kb)
    return f"{least_utilized_bin}_{demand_bin}_{previous_action}"


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
