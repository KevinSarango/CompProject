import json
import os


ACTION_TO_PATH = {
    "0": "upper",
    "1": "lower",
}

CURRENT_STATE = "0"


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


def choose_path(src_ip, dst_ip):
    state = CURRENT_STATE

    if Q_TABLE is not None and state in Q_TABLE:
        action = max(Q_TABLE[state], key=Q_TABLE[state].get)
        return ACTION_TO_PATH[action]

    # Fallback if Q-table is missing.
    dst_num = int(dst_ip.split(".")[-1])
    return "upper" if dst_num % 2 == 0 else "lower"
