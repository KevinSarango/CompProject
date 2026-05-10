import json
import os

from config import (
    DEFAULT_FLOW_SIZE_KB,
    LOAD_BALANCE_LIMIT_KB,
    MEDIUM_FLOW_KB,
    SMALL_FLOW_KB,
    UTILIZATION_DIFF_THRESHOLD,
)


ACTION_TO_PATH = {
    "0": "upper",
    "1": "lower",
}

CURRENT_STATE = "1_1_0"


def load_q_table():
    path = "data/q_table.json"

    if not os.path.exists(path):
        print("[WARN] data/q_table.json not found. Using fallback policy.")
        return None

    with open(path, "r") as f:
        return json.load(f)


Q_TABLE = load_q_table()


def get_utilization_bin(upper_load, lower_load):
    upper_util = upper_load / LOAD_BALANCE_LIMIT_KB
    lower_util = lower_load / LOAD_BALANCE_LIMIT_KB

    diff = upper_util - lower_util

    if diff < -UTILIZATION_DIFF_THRESHOLD:
        return 0

    if diff > UTILIZATION_DIFF_THRESHOLD:
        return 2

    return 1


def get_demand_bin(flow_size_kb):
    if flow_size_kb <= SMALL_FLOW_KB:
        return 0

    if flow_size_kb <= MEDIUM_FLOW_KB:
        return 1

    return 2


def build_state(upper_load, lower_load, flow_size_kb, previous_action):
    utilization_bin = get_utilization_bin(upper_load, lower_load)
    demand_bin = get_demand_bin(flow_size_kb)
    return f"{utilization_bin}_{demand_bin}_{previous_action}"


def set_network_state(state):
    global CURRENT_STATE
    CURRENT_STATE = str(state)


def choose_path(src_ip, dst_ip, state=None):
    selected_state = str(state) if state is not None else CURRENT_STATE

    if Q_TABLE is not None and selected_state in Q_TABLE:
        action = max(Q_TABLE[selected_state], key=Q_TABLE[selected_state].get)
        return ACTION_TO_PATH[action]

    # Fallback if Q-table is missing or state was not trained.
    dst_num = int(dst_ip.split(".")[-1])
    return "upper" if dst_num % 2 == 0 else "lower"
