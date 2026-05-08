import random


class SimpleSDNEnv:
    """
    Lightweight Gym-style environment.

    State:
        0 = balanced
        1 = upper busy
        2 = lower busy

    Actions:
        0 = upper path
        1 = lower path

    Goal:
        Learn to avoid the busier path.
    """

    def __init__(self):
        self.state = 0
        self.steps = 0
        self.max_steps = 100

    def reset(self):
        self.steps = 0
        self.state = random.choice([0, 1, 2])
        return self.state

    def step(self, action):
        self.steps += 1

        upper_load = random.randint(1, 10)
        lower_load = random.randint(1, 10)

        if self.state == 1:
            upper_load += 8
        elif self.state == 2:
            lower_load += 8

        if action == 0:
            selected_load = upper_load
            other_load = lower_load
        else:
            selected_load = lower_load
            other_load = upper_load

        reward = -selected_load

        if selected_load < other_load:
            reward += 5
        else:
            reward -= 5

        next_state = random.choice([0, 1, 2])
        self.state = next_state

        done = self.steps >= self.max_steps

        info = {
            "upper_load": upper_load,
            "lower_load": lower_load,
            "selected_load": selected_load,
        }

        return next_state, reward, done, info
