import csv
import json
import os
import random
import time
from collections import Counter

from config import (
    ACTION_TO_PATH,
    NUM_PATHS,
    PATHS,
    Q_TABLE_FILE,
    STATE_VISITS_CSV_FILE,
    STATE_VISITS_FILE,
    TRAINING_BASE_SEED,
    TRAINING_EPISODE_SEEDS_FILE,
    TRAINING_NUM_FLOWS,
    TRAINING_REWARDS_FILE,
    TRAINING_RUNTIME_FILE,
    TRAINING_STEPS_FILE,
)
from generate_trafpy_demands import generate_demands, save_demands
from sdn_gym_env import SimpleSDNEnv


ACTIONS = list(range(NUM_PATHS))
DEMAND_FILE = "data/trafpy_demands.csv"


def format_seconds(seconds):
    minutes, secs = divmod(float(seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)

    if hours:
        return f"{hours}h {minutes}m {secs:.2f}s"

    if minutes:
        return f"{minutes}m {secs:.2f}s"

    return f"{secs:.2f}s"


def make_episode_seeds(episodes, base_seed):
    """
    Creates one unique, deterministic seed per episode.

    Using the same TRAINING_BASE_SEED and episode count reproduces the exact
    same sequence of per-episode traffic demand traces when retraining.
    """
    rng = random.Random(base_seed)
    seeds = []
    seen = set()

    while len(seeds) < episodes:
        seed = rng.randrange(1, 2**31 - 1)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)

    return seeds


def save_episode_seeds(seeds, base_seed, num_flows):
    os.makedirs(os.path.dirname(TRAINING_EPISODE_SEEDS_FILE) or ".", exist_ok=True)

    with open(TRAINING_EPISODE_SEEDS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed", "base_seed", "num_flows"])

        for episode, seed in enumerate(seeds):
            writer.writerow([episode, seed, base_seed, num_flows])


def write_episode_demands(seed, num_flows, verbose=False):
    demands = generate_demands(num_flows=num_flows, seed=seed)
    save_demands(demands, output_file=DEMAND_FILE, verbose=verbose)
    return demands


def save_training_runtime(start_time, end_time, episodes, visited_states, q_table_size):
    os.makedirs("data", exist_ok=True)
    duration = end_time - start_time

    with open(TRAINING_RUNTIME_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episodes",
            "training_num_flows",
            "training_base_seed",
            "visited_states",
            "q_table_states",
            "q_values",
            "start_time_epoch",
            "end_time_epoch",
            "duration_seconds",
            "duration_human",
        ])
        writer.writerow([
            episodes,
            TRAINING_NUM_FLOWS,
            TRAINING_BASE_SEED,
            visited_states,
            q_table_size,
            q_table_size * NUM_PATHS,
            round(start_time, 6),
            round(end_time, 6),
            round(duration, 6),
            format_seconds(duration),
        ])

    return duration


def build_q_table():
    """
    Preserve the Sebas branch state shape:
        least_utilized_path_bin_demand_bin_previous_action

    least_utilized_path_bin: 0..NUM_PATHS-1
    demand_bin: 0, 1, 2
    previous_action: 0..NUM_PATHS-1
    """
    q_table = {}

    for least_utilized_bin in range(NUM_PATHS):
        for demand_bin in range(3):
            for previous_action in range(NUM_PATHS):
                state = f"{least_utilized_bin}_{demand_bin}_{previous_action}"
                q_table[state] = {
                    str(action): 0.0
                    for action in ACTIONS
                }

    return q_table


def save_state_visits(state_visit_counts):
    os.makedirs("data", exist_ok=True)

    with open(STATE_VISITS_FILE, "w") as f:
        json.dump(dict(sorted(state_visit_counts.items())), f, indent=4)

    with open(STATE_VISITS_CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "visit_count"])

        for state, count in sorted(state_visit_counts.items()):
            writer.writerow([state, count])


def train(episodes=3000, alpha=0.2, gamma=0.9, epsilon=1.0):
    training_start = time.time()

    os.makedirs("data", exist_ok=True)
    episode_seeds = make_episode_seeds(episodes, TRAINING_BASE_SEED)
    save_episode_seeds(
        seeds=episode_seeds,
        base_seed=TRAINING_BASE_SEED,
        num_flows=TRAINING_NUM_FLOWS,
    )

    # Create the first demand file before constructing the environment.
    first_demands = write_episode_demands(
        seed=episode_seeds[0],
        num_flows=TRAINING_NUM_FLOWS,
        verbose=False,
    )

    env = SimpleSDNEnv(demand_file=DEMAND_FILE)
    env.set_demands(first_demands)

    q_table = build_q_table()
    state_visit_counts = Counter()

    with open(TRAINING_REWARDS_FILE, "w", newline="") as reward_file, \
         open(TRAINING_STEPS_FILE, "w", newline="") as step_file:

        reward_writer = csv.writer(reward_file)
        step_writer = csv.writer(step_file)

        reward_writer.writerow([
            "episode",
            "episode_seed",
            "num_flows",
            "total_reward",
            "average_reward",
            "epsilon",
        ])

        step_writer.writerow([
            "episode",
            "episode_seed",
            "step",
            "state",
            "state_visit_count",
            "action",
            "path",
            "flow_id",
            "src",
            "dst",
            "start_time",
            "size_kb",
            "path_loads",
            "selected_load",
            "raw_delay",
            "raw_packet_loss",
            "raw_throughput",
            "normalized_delay",
            "normalized_packet_loss",
            "normalized_throughput",
            "action_impact",
            "imbalance",
            "switching_penalty",
            "reward",
        ])

        for episode, episode_seed in enumerate(episode_seeds):
            if episode == 0:
                demands = first_demands
            else:
                demands = write_episode_demands(
                    seed=episode_seed,
                    num_flows=TRAINING_NUM_FLOWS,
                    verbose=False,
                )
                env.set_demands(demands)

            state = env.reset()
            total_reward = 0.0
            step_count = 0
            done = False

            while not done:
                state_key = str(state)
                state_visit_counts[state_key] += 1

                if state_key not in q_table:
                    q_table[state_key] = {
                        str(action): 0.0
                        for action in ACTIONS
                    }

                if random.random() < epsilon:
                    action = random.choice(ACTIONS)
                else:
                    action = int(max(q_table[state_key], key=q_table[state_key].get))

                next_state, reward, done, info = env.step(action)

                if done and not info:
                    break

                next_state_key = str(next_state)

                if next_state_key not in q_table:
                    q_table[next_state_key] = {
                        str(a): 0.0
                        for a in ACTIONS
                    }

                old_q = q_table[state_key][str(action)]
                best_next_q = max(q_table[next_state_key].values())

                new_q = old_q + alpha * (
                    reward + gamma * best_next_q - old_q
                )

                q_table[state_key][str(action)] = new_q

                total_reward += reward
                step_count += 1

                step_writer.writerow([
                    episode,
                    episode_seed,
                    step_count,
                    state_key,
                    state_visit_counts[state_key],
                    action,
                    ACTION_TO_PATH[str(action)],
                    info["flow_id"],
                    info["src"],
                    info["dst"],
                    info["start_time"],
                    info["size_kb"],
                    info["path_loads"],
                    info["selected_load"],
                    info["raw_delay"],
                    info["raw_packet_loss"],
                    info["raw_throughput"],
                    info["normalized_delay"],
                    info["normalized_packet_loss"],
                    info["normalized_throughput"],
                    info["action_impact"],
                    info["imbalance"],
                    info["switching_penalty"],
                    reward,
                ])

                state = next_state

            average_reward = total_reward / step_count if step_count > 0 else 0.0

            reward_writer.writerow([
                episode,
                episode_seed,
                TRAINING_NUM_FLOWS,
                total_reward,
                average_reward,
                epsilon,
            ])

            if episode % 100 == 0:
                print(
                    f"Episode {episode} | "
                    f"Seed={episode_seed} | "
                    f"Steps={step_count} | "
                    f"Total Reward={total_reward:.3f} | "
                    f"Average Reward={average_reward:.3f} | "
                    f"Epsilon={epsilon:.4f}"
                )

            epsilon = max(0.05, epsilon * 0.995)

    with open(Q_TABLE_FILE, "w") as f:
        json.dump(q_table, f, indent=4)

    save_state_visits(state_visit_counts)

    training_end = time.time()
    training_duration = save_training_runtime(
        start_time=training_start,
        end_time=training_end,
        episodes=episodes,
        visited_states=len(state_visit_counts),
        q_table_size=len(q_table),
    )

    print()
    print("Training complete.")
    print(f"Saved Q-table to: {Q_TABLE_FILE}")
    print(f"Saved rewards to: {TRAINING_REWARDS_FILE}")
    print(f"Saved steps to: {TRAINING_STEPS_FILE}")
    print(f"Saved per-episode seed manifest to: {TRAINING_EPISODE_SEEDS_FILE}")
    print(f"Saved state visits to: {STATE_VISITS_FILE}")
    print(f"Saved state visits CSV to: {STATE_VISITS_CSV_FILE}")
    print(f"Saved training runtime to: {TRAINING_RUNTIME_FILE}")
    print(f"Training runtime: {format_seconds(training_duration)}")
    print()
    print(f"Paths: {PATHS}")
    print(f"Training flows per episode: {TRAINING_NUM_FLOWS}")
    print(f"Training base seed: {TRAINING_BASE_SEED}")
    print(f"Q-table states: {len(q_table)}")
    print(f"Q-values: {len(q_table) * NUM_PATHS}")
    print(f"Visited states: {len(state_visit_counts)}")
    print()
    print("Sample learned states:")

    for state in sorted(q_table.keys())[:10]:
        values = q_table[state]
        best_action = max(values, key=values.get)
        print(
            f"state={state}, "
            f"best_action={ACTION_TO_PATH[best_action]}, "
            f"visits={state_visit_counts.get(state, 0)}, "
            f"values={values}"
        )


if __name__ == "__main__":
    train()
