import csv
import json
import os
import matplotlib.pyplot as plt

PLOTS_DIR = "data/plots"

def ensure_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)

def read_rewards():
    episodes, total_rewards, average_rewards = [], [], []
    with open("data/training_rewards.csv") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            total_rewards.append(float(row["total_reward"]))
            average_rewards.append(float(row["average_reward"]))
    return episodes, total_rewards, average_rewards

def plot_rewards():
    episodes, total_rewards, average_rewards = read_rewards()
    plt.figure()
    plt.plot(episodes, total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("RL Training: Total Reward per Episode")
    plt.savefig(os.path.join(PLOTS_DIR, "reward_curve_total.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(episodes, average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("RL Training: Average Reward per Episode")
    plt.savefig(os.path.join(PLOTS_DIR, "reward_curve_average.png"), bbox_inches="tight")
    plt.close()

def parse_state_key(state_key):
    return tuple(int(part) for part in state_key.split("_"))

def plot_q_table():
    with open("data/q_table.json") as f:
        q_table = json.load(f)
    states = sorted(q_table.keys(), key=parse_state_key)
    upper = [q_table[s]["0"] for s in states]
    lower = [q_table[s]["1"] for s in states]
    x = list(range(len(states)))

    plt.figure(figsize=(14, 6))
    plt.plot(x, upper, marker="o", label="upper path")
    plt.plot(x, lower, marker="o", label="lower path")
    plt.xlabel("Q-table State Index")
    plt.ylabel("Q-value")
    plt.title("Learned Q-table Values Across 54 States")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "q_table_values.png"), bbox_inches="tight")
    plt.close()

    best_actions = [0 if q_table[s]["0"] >= q_table[s]["1"] else 1 for s in states]
    plt.figure(figsize=(14, 4))
    plt.step(x, best_actions, where="mid")
    plt.yticks([0, 1], ["upper", "lower"])
    plt.xlabel("Q-table State Index")
    plt.ylabel("Best Action")
    plt.title("Learned Best Action Per Q-table State")
    plt.savefig(os.path.join(PLOTS_DIR, "q_table_best_actions.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    ensure_directories()
    plot_rewards()
    plot_q_table()
    print("Saved plots:")
    print(f"- {PLOTS_DIR}/reward_curve_total.png")
    print(f"- {PLOTS_DIR}/reward_curve_average.png")
    print(f"- {PLOTS_DIR}/q_table_values.png")
    print(f"- {PLOTS_DIR}/q_table_best_actions.png")
