import csv
import json
import os

import matplotlib.pyplot as plt

from config import (
    ACTION_TO_PATH,
    MAX_POLICY_STATES_TO_PLOT,
    MIN_STATE_VISITS_FOR_POLICY_PLOT,
    PATHS,
    PLOTS_DIR,
    PLOT_PREFIX,
    Q_TABLE_FILE,
    STATE_VISITS_FILE,
    TRAINING_REWARDS_FILE,
)


def ensure_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def read_rewards():
    episodes = []
    total_rewards = []
    average_rewards = []

    with open(TRAINING_REWARDS_FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            episodes.append(int(row["episode"]))
            total_rewards.append(float(row["total_reward"]))
            average_rewards.append(float(row["average_reward"]))

    return episodes, total_rewards, average_rewards


def plot_rewards():
    episodes, total_rewards, average_rewards = read_rewards()

    plt.figure()
    plt.plot(episodes, total_rewards, label="Total Reward")

    if total_rewards:
        mean_total_reward = sum(total_rewards) / len(total_rewards)
        plt.axhline(
            mean_total_reward,
            linestyle="--",
            linewidth=1.5,
            label=f"Mean Total Reward = {mean_total_reward:.3f}",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"RL Training: Total Reward per Episode ({PLOT_PREFIX})")
    plt.legend()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_reward_curve_total.png"),
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(episodes, average_rewards, label="Average Reward")

    if average_rewards:
        mean_average_reward = sum(average_rewards) / len(average_rewards)
        plt.axhline(
            mean_average_reward,
            linestyle="--",
            linewidth=1.5,
            label=f"Mean Average Reward = {mean_average_reward:.3f}",
        )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"RL Training: Average Reward per Episode ({PLOT_PREFIX})")
    plt.legend()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_reward_curve_average.png"),
        bbox_inches="tight",
    )
    plt.close()


def sorted_states(q_table):
    return sorted(
        q_table.keys(),
        key=lambda s: tuple(int(part) for part in s.split("_")),
    )


def load_state_visits():
    if not os.path.exists(STATE_VISITS_FILE):
        print(f"[WARN] Missing state visits file: {STATE_VISITS_FILE}")
        return {}

    with open(STATE_VISITS_FILE, "r") as f:
        return json.load(f)


def filter_visited_states(q_table, state_visits):
    if not state_visits:
        return sorted_states(q_table)

    states = [
        state
        for state in q_table.keys()
        if int(state_visits.get(state, 0)) >= MIN_STATE_VISITS_FOR_POLICY_PLOT
    ]

    states.sort(
        key=lambda s: (
            -int(state_visits.get(s, 0)),
            tuple(int(part) for part in s.split("_")),
        )
    )

    return states[:MAX_POLICY_STATES_TO_PLOT]


def plot_q_table_policy():
    with open(Q_TABLE_FILE, "r") as f:
        q_table = json.load(f)

    state_visits = load_state_visits()
    states = filter_visited_states(q_table, state_visits)

    best_actions = []

    for state in states:
        values = q_table[state]
        best_action = int(max(values, key=values.get))
        best_actions.append(best_action)

    x = range(len(states))

    plt.figure(figsize=(max(12, len(states) * 0.18), 5))
    plt.plot(list(x), best_actions, marker="o", linewidth=1)
    plt.yticks(
        list(range(len(PATHS))),
        [ACTION_TO_PATH[str(i)] for i in range(len(PATHS))],
    )
    plt.xticks(list(x), states, rotation=90)
    plt.xlabel(
        "Visited state: least_utilized_path_demand_bin_previous_action "
        f"(min visits={MIN_STATE_VISITS_FOR_POLICY_PLOT})"
    )
    plt.ylabel("Best Action")
    plt.title(f"Learned Policy from Visited Q-table States ({PLOT_PREFIX})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_q_table_policy.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_q_table_values():
    with open(Q_TABLE_FILE, "r") as f:
        q_table = json.load(f)

    state_visits = load_state_visits()
    states = filter_visited_states(q_table, state_visits)
    x = range(len(states))

    plt.figure(figsize=(max(14, len(states) * 0.2), 6))

    width = 0.8 / len(PATHS)

    for action_index, path_name in enumerate(PATHS):
        values = [q_table[s][str(action_index)] for s in states]
        offsets = [i - 0.4 + width / 2 + action_index * width for i in x]

        plt.bar(
            offsets,
            values,
            width=width,
            label=path_name,
        )

    plt.xticks(list(x), states, rotation=90)
    plt.xlabel("Visited state: least_utilized_path_demand_bin_previous_action")
    plt.ylabel("Q-value")
    plt.title(f"Q-table Values for Visited States ({PLOT_PREFIX})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_q_table_values.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_state_visit_counts():
    state_visits = load_state_visits()

    if not state_visits:
        return

    items = sorted(
        state_visits.items(),
        key=lambda item: (-int(item[1]), tuple(int(part) for part in item[0].split("_"))),
    )[:MAX_POLICY_STATES_TO_PLOT]

    states = [item[0] for item in items]
    counts = [int(item[1]) for item in items]
    x = range(len(states))

    plt.figure(figsize=(max(12, len(states) * 0.18), 5))
    plt.bar(list(x), counts)
    plt.xticks(list(x), states, rotation=90)
    plt.xlabel("Visited state")
    plt.ylabel("Visit Count")
    plt.title(f"Most Visited Training States ({PLOT_PREFIX})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_state_visit_counts.png"),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    ensure_directories()
    plot_rewards()
    plot_q_table_values()
    plot_q_table_policy()
    plot_state_visit_counts()

    print("Saved plots:")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_reward_curve_total.png")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_reward_curve_average.png")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_q_table_values.png")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_q_table_policy.png")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_state_visit_counts.png")
