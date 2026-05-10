import csv
import json
import os
import random

from sdn_gym_env import SimpleSDNEnv


ACTIONS = [0, 1]

ACTION_NAMES = {
    0: "upper",
    1: "lower",
}


def train(episodes=3000, alpha=0.2, gamma=0.9, epsilon=1.0):
    env = SimpleSDNEnv()

    q_table = {
        "0": {"0": 0.0, "1": 0.0},
        "1": {"0": 0.0, "1": 0.0},
        "2": {"0": 0.0, "1": 0.0},
        "3": {"0": 0.0, "1": 0.0},
    }

    os.makedirs("data", exist_ok=True)

    with open("data/training_rewards.csv", "w", newline="") as reward_file, \
         open("data/training_steps.csv", "w", newline="") as step_file:

        reward_writer = csv.writer(reward_file)
        step_writer = csv.writer(step_file)

        reward_writer.writerow([
            "episode",
            "total_reward",
            "average_reward",
            "epsilon",
        ])

        step_writer.writerow([
            "episode",
            "step",
            "state",
            "action",
            "path",
            "flow_id",
            "src",
            "dst",
            "start_time",
            "size_kb",
            "upper_load",
            "lower_load",
            "selected_load",
            "raw_delay",
            "raw_packet_loss",
            "raw_throughput",
            "normalized_delay",
            "normalized_packet_loss",
            "normalized_throughput",
            "action_impact",
            "reward",
        ])

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0.0
            step_count = 0
            done = False

            while not done:
                state_key = str(state)

                # Epsilon-greedy action selection.
                if random.random() < epsilon:
                    action = random.choice(ACTIONS)
                else:
                    action = int(max(q_table[state_key], key=q_table[state_key].get))

                next_state, reward, done, info = env.step(action)

                if done and not info:
                    break

                next_state_key = str(next_state)

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
                    step_count,
                    state_key,
                    action,
                    ACTION_NAMES[action],
                    info["flow_id"],
                    info["src"],
                    info["dst"],
                    info["start_time"],
                    info["size_kb"],
                    info["upper_load"],
                    info["lower_load"],
                    info["selected_load"],
                    info["raw_delay"],
                    info["raw_packet_loss"],
                    info["raw_throughput"],
                    info["normalized_delay"],
                    info["normalized_packet_loss"],
                    info["normalized_throughput"],
                    info["action_impact"],
                    reward,
                ])

                state = next_state

            average_reward = total_reward / step_count if step_count > 0 else 0.0

            reward_writer.writerow([
                episode,
                total_reward,
                average_reward,
                epsilon,
            ])

            if episode % 100 == 0:
                print(
                    f"Episode {episode} | "
                    f"Steps={step_count} | "
                    f"Total Reward={total_reward:.3f} | "
                    f"Average Reward={average_reward:.3f} | "
                    f"Epsilon={epsilon:.4f}"
                )

            # Epsilon decay: more exploration early, more exploitation later.
            epsilon = max(0.05, epsilon * 0.995)

    with open("data/q_table.json", "w") as f:
        json.dump(q_table, f, indent=4)

    print()
    print("Training complete.")
    print("Saved:")
    print("- data/q_table.json")
    print("- data/training_rewards.csv")
    print("- data/training_steps.csv")
    print()
    print("Final Q-table:")

    for state, values in q_table.items():
        best_action = max(values, key=values.get)
        print(
            f"state={state}, "
            f"best_action={ACTION_NAMES[int(best_action)]}, "
            f"values={values}"
        )


if __name__ == "__main__":
    train()
