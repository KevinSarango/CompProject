import random


class SimpleSDNEnv:
    """
    Gym-style SDN multipath environment.

    Actions:
        0 = upper path: s1 -> s2 -> s4
        1 = lower path: s1 -> s3 -> s4

    States:
        0 = balanced traffic
        1 = upper path congested
        2 = lower path congested
        3 = both paths busy

    Reward inspired by SDN-RL literature:
        reward =
            - gamma1 * normalized_packet_loss
            - gamma2 * normalized_delay
            + gamma3 * normalized_throughput
            + gamma4 * action_impact

    Static weights:
        gamma1 = 2.0   packet loss
        gamma2 = 1.5   delay
        gamma3 = 1.0   throughput
        gamma4 = 1.0   action impact / causal-style influence
    """

    def __init__(self):
        self.state = 0
        self.steps = 0
        self.max_steps = 100

        # Reward weights
        self.gamma_packet_loss = 2.0
        self.gamma_delay = 1.5
        self.gamma_throughput = 1.0
        self.gamma_action_impact = 1.0

        # Normalization constants.
        # These are based on the simulated ranges in this environment.
        self.max_load = 25.0
        self.max_delay = 50.0
        self.max_packet_loss = 13.0
        self.max_throughput = 25.0

    def reset(self):
        self.steps = 0
        self.state = random.choice([0, 1, 2, 3])
        return self.state

    def generate_loads(self):
        upper_load = random.randint(1, 10)
        lower_load = random.randint(1, 10)

        if self.state == 1:
            upper_load += random.randint(8, 15)
        elif self.state == 2:
            lower_load += random.randint(8, 15)
        elif self.state == 3:
            upper_load += random.randint(5, 12)
            lower_load += random.randint(5, 12)

        return upper_load, lower_load

    def normalize(self, value, max_value):
        if max_value == 0:
            return 0.0

        value = max(0.0, min(float(value), float(max_value)))
        return value / max_value

    def calculate_action_impact(self, selected_load, other_load):
        """
        Action impact approximates the causal influence term.

        Since this project does not implement full causal inference, we define
        action impact as how much better the selected path is compared to the
        alternative path.

        Positive impact:
            selected path is less congested than other path

        Negative impact:
            selected path is more congested than other path
        """

        load_difference = other_load - selected_load

        # Normalize to approximately [-1, 1]
        action_impact = load_difference / self.max_load

        if action_impact > 1.0:
            action_impact = 1.0
        elif action_impact < -1.0:
            action_impact = -1.0

        return action_impact

    def step(self, action):
        self.steps += 1

        upper_load, lower_load = self.generate_loads()

        if action == 0:
            selected_path = "upper"
            selected_load = upper_load
            other_load = lower_load
        else:
            selected_path = "lower"
            selected_load = lower_load
            other_load = upper_load

        # Simulated raw network metrics
        raw_delay = selected_load * 2.0
        raw_packet_loss = max(0.0, selected_load - 12.0)
        raw_throughput = max(0.0, self.max_throughput - selected_load)

        # Normalized metrics in range [0, 1]
        normalized_delay = self.normalize(raw_delay, self.max_delay)
        normalized_packet_loss = self.normalize(raw_packet_loss, self.max_packet_loss)
        normalized_throughput = self.normalize(raw_throughput, self.max_throughput)

        # Approximate causal/action influence in range [-1, 1]
        action_impact = self.calculate_action_impact(
            selected_load=selected_load,
            other_load=other_load,
        )

        reward = (
            -self.gamma_packet_loss * normalized_packet_loss
            -self.gamma_delay * normalized_delay
            +self.gamma_throughput * normalized_throughput
            +self.gamma_action_impact * action_impact
        )

        next_state = random.choice([0, 1, 2, 3])
        self.state = next_state

        done = self.steps >= self.max_steps

        info = {
            "state": self.state,
            "selected_path": selected_path,
            "upper_load": upper_load,
            "lower_load": lower_load,
            "selected_load": selected_load,
            "raw_delay": raw_delay,
            "raw_packet_loss": raw_packet_loss,
            "raw_throughput": raw_throughput,
            "normalized_delay": normalized_delay,
            "normalized_packet_loss": normalized_packet_loss,
            "normalized_throughput": normalized_throughput,
            "action_impact": action_impact,
            "reward": reward,
        }

        return next_state, reward, done, info
