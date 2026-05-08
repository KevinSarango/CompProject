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


def train(episodes=1000, alpha=0.2, gamma=0.9, epsilon=0.2):
    env = SimpleSDNEnv()

    q_table = {
        "0": {"0": 0.0, "1": 0.0},
        "1": {"0": 0.0, "1": 0.0},
        "2": {"0": 0.0, "1": 0.0},
    }

    os.makedirs("data", exist_ok=True)

    with open("data/training_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward"])

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            done = False
            while not done:
                state_key = str(state)

                if random.random() < epsilon:
                    action = random.choice(ACTIONS)
                else:
                    action_values = q_table[state_key]
                    action = int(max(action_values, key=action_values.get))

                next_state, reward, done, info = env.step(action)
                next_state_key = str(next_state)

                old_q = q_table[state_key][str(action)]
                best_next_q = max(q_table[next_state_key].values())

                new_q = old_q + alpha * (
                    reward + gamma * best_next_q - old_q
                )

                q_table[state_key][str(action)] = new_q
                total_reward += reward
                state = next_state

            writer.writerow([episode, total_reward])

    with open("data/q_table.json", "w") as f:
        json.dump(q_table, f, indent=4)

    print("Training complete.")
    print("Saved Q-table to data/q_table.json")
    print("Saved rewards to data/training_rewards.csv")

    for state, values in q_table.items():
        best_action = max(values, key=values.get)
        print(f"state={state}, best_action={ACTION_NAMES[int(best_action)]}, values={values}")


if __name__ == "__main__":
    train()
