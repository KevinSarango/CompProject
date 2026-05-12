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
    plt.plot(episodes, total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"RL Training: Total Reward per Episode ({PLOT_PREFIX})")
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_reward_curve_total.png"),
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(episodes, average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"RL Training: Average Reward per Episode ({PLOT_PREFIX})")
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_reward_curve_average.png"),
        bbox_inches="tight",
    )
    plt.close()


def sorted_states(states):
    return sorted(
        states,
        key=lambda s: tuple(int(part) for part in s.split("_")),
    )


def load_q_table():
    with open(Q_TABLE_FILE, "r") as f:
        return json.load(f)


def load_state_visits():
    if not os.path.exists(STATE_VISITS_FILE):
        print(f"[WARN] {STATE_VISITS_FILE} not found. Plotting all Q-table states.")
        return None

    with open(STATE_VISITS_FILE, "r") as f:
        return {state: int(count) for state, count in json.load(f).items()}


def get_plotted_states(q_table, visits):
    if visits is None:
        states = sorted_states(q_table.keys())
        return states[:MAX_POLICY_STATES_TO_PLOT]

    visited_states = [
        state
        for state, count in visits.items()
        if count >= MIN_STATE_VISITS_FOR_POLICY_PLOT and state in q_table
    ]

    # Plot the most frequently visited states first. This avoids large policy
    # plots being dominated by rare/unimportant states.
    visited_states.sort(
        key=lambda state: (-visits[state], tuple(int(part) for part in state.split("_")))
    )

    return visited_states[:MAX_POLICY_STATES_TO_PLOT]


def plot_state_visit_counts():
    visits = load_state_visits()

    if not visits:
        return

    counts = sorted(visits.values(), reverse=True)
    x = range(len(counts))

    plt.figure(figsize=(12, 5))
    plt.plot(list(x), counts)
    plt.xlabel("Visited state rank")
    plt.ylabel("Visit count")
    plt.title(f"State Visit Counts ({PLOT_PREFIX})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_state_visit_counts.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_q_table_policy():
    q_table = load_q_table()
    visits = load_state_visits()
    states = get_plotted_states(q_table, visits)

    best_actions = []
    labels = []

    for state in states:
        values = q_table[state]
        best_action = int(max(values, key=values.get))
        best_actions.append(best_action)
        visit_suffix = f"\nvisits={visits[state]}" if visits is not None else ""
        labels.append(f"{state}{visit_suffix}")

    x = range(len(states))

    plt.figure(figsize=(14, 5))
    plt.plot(list(x), best_actions, marker="o")
    plt.yticks(
        list(range(len(PATHS))),
        [ACTION_TO_PATH[str(i)] for i in range(len(PATHS))],
    )
    if len(states) <= 80:
        plt.xticks(list(x), labels, rotation=90)
    else:
        plt.xticks([])
    plt.xlabel(
        f"Visited states only; min visits={MIN_STATE_VISITS_FOR_POLICY_PLOT}; "
        f"showing up to {MAX_POLICY_STATES_TO_PLOT}"
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
    q_table = load_q_table()
    visits = load_state_visits()
    states = get_plotted_states(q_table, visits)
    x = range(len(states))

    plt.figure(figsize=(14, 6))

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

    if len(states) <= 80:
        labels = [
            f"{state}\nvisits={visits[state]}" if visits is not None else state
            for state in states
        ]
        plt.xticks(list(x), labels, rotation=90)
    else:
        plt.xticks([])
    plt.xlabel(
        f"Visited states only; min visits={MIN_STATE_VISITS_FOR_POLICY_PLOT}; "
        f"showing up to {MAX_POLICY_STATES_TO_PLOT}"
    )
    plt.ylabel("Q-value")
    plt.title(f"Q-table Values for Visited States ({PLOT_PREFIX})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_q_table_values.png"),
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
