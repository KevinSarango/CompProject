import json
import os
from config import LOAD_BALANCE_LIMIT_KB, UTIL_DIFF_THRESHOLD, DELAY_DIFF_THRESHOLD, SMALL_FLOW_KB, LARGE_FLOW_KB

ACTION_TO_PATH = {"0": "upper", "1": "lower"}
CURRENT_STATE = "1_1_1_0"

def load_q_table():
    path = "data/q_table.json"
    if not os.path.exists(path):
        print("[WARN] data/q_table.json not found. Using fallback policy.")
        return None
    with open(path, "r") as f:
        return json.load(f)

Q_TABLE = load_q_table()

def set_network_state(state):
    global CURRENT_STATE
    CURRENT_STATE = str(state)

def demand_bin(size_kb):
    if size_kb <= SMALL_FLOW_KB:
        return 0
    if size_kb <= LARGE_FLOW_KB:
        return 1
    return 2

def difference_bin(upper_value, lower_value, threshold):
    diff = upper_value - lower_value
    if diff < -threshold:
        return 0
    if diff > threshold:
        return 2
    return 1

def build_state_key(upper_load, lower_load, flow_size_kb, previous_action):
    upper_util = upper_load / LOAD_BALANCE_LIMIT_KB
    lower_util = lower_load / LOAD_BALANCE_LIMIT_KB
    util_bin = difference_bin(upper_util, lower_util, UTIL_DIFF_THRESHOLD)
    delay_bin = difference_bin(upper_util, lower_util, DELAY_DIFF_THRESHOLD)
    size_bin = demand_bin(flow_size_kb)
    return f"{util_bin}_{delay_bin}_{size_bin}_{int(previous_action)}"

def choose_path(src_ip, dst_ip, state_key=None):
    state = str(state_key) if state_key is not None else CURRENT_STATE
    if Q_TABLE is not None and state in Q_TABLE:
        action = max(Q_TABLE[state], key=Q_TABLE[state].get)
        return ACTION_TO_PATH[action]
    dst_num = int(dst_ip.split(".")[-1])
    return "upper" if dst_num % 2 == 0 else "lower"
