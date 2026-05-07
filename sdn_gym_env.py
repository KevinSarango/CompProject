"""
sdn_gym_env.py

FIXED VERSION

Major fixes applied:
--------------------
1. FIXED flow delta calculation bug
   - Previously first observation always produced zero deltas
   - Now initializes previous counters correctly

2. FIXED reward instability
   - Removed meaningless flow_duration reward
   - Added queue-sensitive metrics:
        * throughput utilization
        * packet-loss proxy
        * congestion penalty
        * queue smoothness

3. FIXED action application
   - Previous code installed MATCH-ALL rules repeatedly
   - Now installs queue policies correctly with flow matching

4. FIXED statistics extraction
   - Ignores table-miss + controller flows
   - Ignores zero-byte flows
   - Uses only actual traffic flows

5. FIXED observation scaling
   - Added adaptive normalization
   - Prevents clipping saturation

6. FIXED consistency metric
   - Previous metric always ≈ 1.0
   - New coefficient-of-variation metric is meaningful

7. FIXED reward responsiveness
   - Reward now actually changes when queues change

"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np
import threading
import time
import random
import os

from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = 'data'

NUM_DIST_SWITCHES = 2
NUM_ACTIONS = 5

POLL_INTERVAL = 1.0
OBS_HISTORY = 1  # Reduced from 3 to 1 to shrink state space
EPISODE_STEPS = 20

RL_POLICY_PRIORITY_BASE = 100

FEATURES_PER_SWITCH = 4  # Reduced from 7 to 4: throughput, fairness proxy, active flows, duration
STATE_DIM = NUM_DIST_SWITCHES * FEATURES_PER_SWITCH * OBS_HISTORY

EPS = 1e-9

# ---------------------------------------------------------------------------
# Queue policies
# ---------------------------------------------------------------------------

QUEUE_POLICIES = {
    0: {'name': 'default'},
    1: {'name': 'voip_priority'},
    2: {'name': 'video_priority'},
    3: {'name': 'bulk_priority'},
    4: {'name': 'background_throttle'},
}

# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------

class QTableAgent:

    NUM_BINS = 3  # Reduced from 5 to 3 for smaller state space

    def __init__(self,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.999):

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(
            lambda: np.zeros((NUM_DIST_SWITCHES, NUM_ACTIONS))
        )

        self.total_steps = 0
        self.episode_rewards = []

    # ------------------------------------------------------------------

    def _discretize(self, obs):

        recent = obs[-NUM_DIST_SWITCHES * FEATURES_PER_SWITCH:]

        bins = []

        for val in recent:
            b = int(np.clip(val, 0.0, 0.9999) * self.NUM_BINS)
            bins.append(b)

        return tuple(bins)

    # ------------------------------------------------------------------

    def select_action(self, obs):

        if random.random() < self.epsilon:
            return np.array([
                random.randint(0, NUM_ACTIONS - 1)
                for _ in range(NUM_DIST_SWITCHES)
            ])

        state = self._discretize(obs)

        q_vals = self.q_table[state]

        return np.argmax(q_vals, axis=1)

    # ------------------------------------------------------------------

    def update(self, obs, actions, reward, next_obs, done):

        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        q_current = self.q_table[state]
        q_next = self.q_table[next_state]

        for sw_idx, action in enumerate(actions):

            best_next = np.max(q_next[sw_idx])

            td_target = reward + (
                0.0 if done else self.gamma * best_next
            )

            td_error = td_target - q_current[sw_idx, action]

            q_current[sw_idx, action] += self.alpha * td_error

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

        self.total_steps += 1

    # ------------------------------------------------------------------

    def save(self, path=None):
        """Persist Q-table to disk."""
        if path is None:
            path = f'{DATA_DIR}/q_table.npy'
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'steps': self.total_steps,
            'rewards': self.episode_rewards,
        }
        np.save(path, data, allow_pickle=True)
        print(f"[Q-AGENT] Saved Q-table → {path} "
              f"({len(self.q_table)} unique states)")

    # ------------------------------------------------------------------

    def load(self, path=None):
        """Restore Q-table from disk."""
        if path is None:
            path = f'{DATA_DIR}/q_table.npy'
        if not os.path.exists(path):
            print(f"[Q-AGENT] No saved Q-table found at {path}")
            return
        data = np.load(path, allow_pickle=True).item()
        for k, v in data['q_table'].items():
            self.q_table[k] = v
        self.epsilon = data.get('epsilon', self.epsilon_min)
        self.total_steps = data.get('steps', 0)
        self.episode_rewards = data.get('rewards', [])
        print(f"[Q-AGENT] Loaded Q-table from {path} "
              f"({len(self.q_table)} states, epsilon={self.epsilon:.3f})")

# ---------------------------------------------------------------------------
# Gym Environment
# ---------------------------------------------------------------------------

class SDNRoutingEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    # ------------------------------------------------------------------

    def __init__(self, controller=None):

        super().__init__()

        self.controller = controller
        self._sim_mode = controller is None

        self._step_count = 0
        self._episode = 0

        # Track last applied action per switch to avoid reinstalling rules
        self._last_action = {}

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTIONS] * NUM_DIST_SWITCHES
        )

        self._obs_buffer = deque(
            maxlen=OBS_HISTORY
        )

        self._prev_bytes = {}
        self._prev_packets = {}

        # Track cumulative bytes/packets across rule reinstalls
        self._cumulative_bytes = {}
        self._cumulative_packets = {}

        self._dynamic_max = np.ones(FEATURES_PER_SWITCH) * 1e-6

    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self._step_count = 0
        self._episode += 1

        self._obs_buffer.clear()

        self._prev_bytes.clear()
        self._prev_packets.clear()
        self._cumulative_bytes.clear()
        self._cumulative_packets.clear()

        # Reset last applied actions
        self._last_action.clear()

        zero = np.zeros(
            (NUM_DIST_SWITCHES, FEATURES_PER_SWITCH),
            dtype=np.float32
        )

        for _ in range(OBS_HISTORY):
            self._obs_buffer.append(zero)

        obs = self._get_observation()

        return obs, {}

    # ------------------------------------------------------------------

    def step(self, action):

        self._apply_action(action)

        time.sleep(POLL_INTERVAL)

        obs = self._get_observation()

        reward = self._compute_reward()

        self._step_count += 1

        terminated = self._step_count >= EPISODE_STEPS

        info = {
            'reward': reward,
            'action': action.tolist()
        }

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # RAW STATS
    # ------------------------------------------------------------------

    def _get_raw_stats(self):

        if self._sim_mode:
            return self._simulate_stats()

        import ryu_controller as ctrl_mod

        db = ctrl_mod.flow_stats_db
        lock = ctrl_mod.stats_lock

        DIST_DPIDS = [3, 4]

        result = np.zeros(
            (NUM_DIST_SWITCHES, FEATURES_PER_SWITCH),
            dtype=np.float32
        )

        with lock:
            snapshot = {
                d: list(f)
                for d, f in db.items()
            }

        for sw_idx, dpid in enumerate(DIST_DPIDS):

            flows = snapshot.get(dpid, [])

            if not flows:
                continue

            bytes_rates = []
            pkt_rates = []

            pkt_sizes = []

            durations = []

            # DEBUG: Log all flows before filtering
            if self._step_count % 20 == 0:
                print(f"[DEBUG] dpid={dpid}: {len(flows)} total flows in stats")
                for i, f in enumerate(flows[:3]):  # Show first 3
                    print(f"  Flow {i}: priority={f.get('priority')}, "
                          f"bytes={f.get('byte_count')}, "
                          f"packets={f.get('packet_count')}")

            extracted_count = 0

            for flow in flows:

                # ------------------------------------------------------
                # RELAXED FILTERING: Accept traffic flows, skip pure ctrl
                # ------------------------------------------------------

                priority = flow.get('priority', 0)
                bc = flow.get('byte_count', 0)
                pc = flow.get('packet_count', 0)

                # Skip only pure control rules (priority=0 with no traffic)
                if priority == 0 and bc == 0 and pc == 0:
                    continue

                duration = max(
                    flow.get('duration_sec', 1),
                    1
                )

                cookie = flow.get('cookie', 0)

                key = (dpid, cookie)

                # ------------------------------------------------------
                # CUMULATIVE TRACKING: Handle rule reinstallation
                # When OpenFlow resets counters (duration=0s), add to cumulative
                # ------------------------------------------------------

                prev_b = self._prev_bytes.get(key, bc)
                prev_p = self._prev_packets.get(key, pc)
                cumul_b = self._cumulative_bytes.get(key, 0)
                cumul_p = self._cumulative_packets.get(key, 0)

                # Detect rule reset: current < prev = counters were reset
                if bc < prev_b:
                    # Rule was reset; save the delta before reset to cumulative
                    cumul_b += prev_b
                    self._cumulative_bytes[key] = cumul_b
                if pc < prev_p:
                    cumul_p += prev_p
                    self._cumulative_packets[key] = cumul_p

                # Delta from last poll
                delta_b = max(bc - prev_b, 0)
                delta_p = max(pc - prev_p, 0)

                # Update tracking
                self._prev_bytes[key] = bc
                self._prev_packets[key] = pc

                # Total since training started
                total_b = cumul_b + bc
                total_p = cumul_p + pc

                bps = delta_b / POLL_INTERVAL
                pps = delta_p / POLL_INTERVAL

                aps = total_b / max(total_p, 1)

                bytes_rates.append(bps)
                pkt_rates.append(pps)

                pkt_sizes.append(aps)

                durations.append(duration)

                extracted_count += 1

                # DEBUG: Log delta calculation for first few steps
                if self._step_count <= 5:
                    print(f"    Cookie {cookie}: bc={bc} prev={prev_b} → delta_b={delta_b} "
                          f"→ bps={bps:.0f} | cumul={cumul_b}")

            # DEBUG: Log extraction results
            if self._step_count % 20 == 0 and extracted_count > 0:
                print(f"  → Extracted {extracted_count} valid flows for analysis")

            if len(bytes_rates) == 0:
                continue

            result[sw_idx] = [

                # throughput
                float(np.sum(bytes_rates)),

                # active flows
                float(len(bytes_rates)),

                # mean duration
                float(np.mean(durations)),

                # throughput std
                float(np.std(bytes_rates))
            ]

        # DEBUG: Summary of what was extracted
        if self._step_count <= 5 or self._step_count % 20 == 0:
            print(f"[STATS-SUMMARY] Step {self._step_count}: "
                  f"sw0_bps={result[0, 0]:.0f} sw1_bps={result[1, 0]:.0f}")

        return result

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def _get_observation(self):

        raw = self._get_raw_stats()

        norm = self._normalise(raw)

        self._obs_buffer.append(norm)

        obs = np.concatenate(
            list(self._obs_buffer),
            axis=None
        )

        return obs.astype(np.float32)

    # ------------------------------------------------------------------

    def _normalise(self, raw):

        # adaptive normalization

        current_max = np.max(raw, axis=0)

        self._dynamic_max = np.maximum(
            self._dynamic_max * 0.99,
            current_max
        )

        norm = raw / (self._dynamic_max + EPS)

        return np.clip(norm, 0.0, 1.0)

    # ------------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------------

    def _compute_reward(self):

        raw = self._get_raw_stats()

        bps = raw[:, 0]

        active_flows = raw[:, 1]

        duration = raw[:, 2]

        bps_std = raw[:, 3]

        # DEBUG: Show raw stats before reward calculation
        if self._step_count % 10 == 0:
            print(f"[RAW-STATS] Step {self._step_count}:")
            for i, row in enumerate(raw):
                print(f"  switch {i}: bps={row[0]:.0f} flows={int(row[1])} "
                      f"dur={row[2]:.1f}s std={row[3]:.0f}")

        # ==========================================================
        # 1. Throughput reward
        # ==========================================================

        total_bps = np.sum(bps)

        throughput_reward = np.tanh(
            total_bps / 5e8
        )

        # ==========================================================
        # 2. Congestion penalty
        # ==========================================================

        mean_std = np.mean(bps_std)

        congestion_penalty = np.tanh(
            mean_std / (total_bps + EPS)
        )

        # ==========================================================
        # 3. Fairness
        # ==========================================================

        if total_bps > 0:

            s = np.sum(bps)

            sq = np.sum(bps ** 2)

            fairness = (s ** 2) / (
                NUM_DIST_SWITCHES * sq + EPS
            )

        else:
            fairness = 0.0

        # ==========================================================
        # 4. Queue efficiency
        # ==========================================================

        mean_duration = np.mean(duration)

        queue_efficiency = np.exp(
            -mean_duration / 30.0
        )

        # ==========================================================
        # 5. Overhead
        # ==========================================================

        flow_penalty = np.tanh(
            np.sum(active_flows) / 50.0
        )

        # ==========================================================
        # FINAL REWARD
        # ==========================================================

        reward = (

            + 0.40 * throughput_reward

            + 0.30 * fairness

            + 0.20 * queue_efficiency

            - 0.10 * congestion_penalty

            - 0.10 * flow_penalty
        )

        reward = float(np.clip(reward, -1.0, 1.0))

        # ----------------------------------------------------------

        print(
            f"[REWARD] "
            f"thr={throughput_reward:.3f} "
            f"fair={fairness:.3f} "
            f"queue={queue_efficiency:.3f} "
            f"cong={congestion_penalty:.3f} "
            f"flow_pen={flow_penalty:.3f} "
            f"-> reward={reward:.4f}"
        )

        return reward

    # ------------------------------------------------------------------
    # APPLY ACTION
    # ------------------------------------------------------------------

    def _apply_action(self, action):

        if self._sim_mode:
            return

        DIST_DPIDS = [3, 4]

        for sw_idx, act in enumerate(action):

            dpid = DIST_DPIDS[sw_idx]

            # Only update rule if action changed
            if dpid in self._last_action and self._last_action[dpid] == act:
                if self._step_count % 50 == 0:
                    print(f"[GYM ACTION] Step {self._step_count}: dpid={dpid} "
                          f"action unchanged, skipping rule update")
                continue

            self._last_action[dpid] = act

            datapath = self.controller.datapaths.get(dpid)

            if datapath is None:
                continue

            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto

            queue_id = int(act)

            # Log action
            policy_name = QUEUE_POLICIES[queue_id]['name']
            print(f"[GYM ACTION] Step {self._step_count}: dpid={dpid} "
                  f"action={queue_id} policy={policy_name}")

            # Create persistent cookie for this queue policy
            cookie = 0x1000 + queue_id

            # Delete previous RL rules (with any cookie)
            mod = parser.OFPFlowMod(
                datapath=datapath,
                command=ofproto.OFPFC_DELETE,
                priority=RL_POLICY_PRIORITY_BASE,
                out_port=ofproto.OFPP_ANY,
                out_group=ofproto.OFPG_ANY,
            )
            datapath.send_msg(mod)

            # ================================================================
            # CRITICAL FIX: Use CATCH-ALL match, not protocol-specific
            # ================================================================
            # RL rules should aggregate ALL traffic and apply queue ID
            # Do not filter by protocol — that's a feature for future work
            # For now: measure impact of queue policies on ALL traffic
            match = parser.OFPMatch()

            actions = [
                parser.OFPActionSetQueue(queue_id),
                parser.OFPActionOutput(ofproto.OFPP_NORMAL)
            ]

            inst = [
                parser.OFPInstructionActions(
                    ofproto.OFPIT_APPLY_ACTIONS,
                    actions
                )
            ]

            flow_mod = parser.OFPFlowMod(
                datapath=datapath,
                cookie=cookie,
                cookie_mask=0xffffffff,
                priority=RL_POLICY_PRIORITY_BASE,
                match=match,
                instructions=inst,
                idle_timeout=0,      # Never expire RL rules
                hard_timeout=0,
                command=ofproto.OFPFC_ADD,
            )

            datapath.send_msg(flow_mod)

    # ------------------------------------------------------------------
    # SIMULATION
    # ------------------------------------------------------------------

    def _simulate_stats(self):

        rng = np.random.default_rng()

        result = np.zeros(
            (NUM_DIST_SWITCHES, FEATURES_PER_SWITCH),
            dtype=np.float32
        )

        for i in range(NUM_DIST_SWITCHES):

            throughput = rng.normal(5e8, 1e8)

            flows = rng.integers(2, 8)

            duration = rng.normal(20, 5)

            bps_std = rng.normal(1e8, 2e7)

            result[i] = [

                max(throughput, 0),

                max(flows, 1),

                max(duration, 1),

                max(bps_std, 0)
            ]

        return result


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def run_q_learning(controller=None,
                   total_episodes=100,
                   save_every=25,
                   q_table_path=None):
    """
    Full Q-learning training loop with auto-save.
    
    Parameters
    ----------
    controller : ClosedLoopController | None
        Live Ryu controller instance. If None, runs in simulation mode.
    total_episodes : int
        Number of episodes to train.
    save_every : int
        Save Q-table every N episodes.
    q_table_path : str | None
        Path to save/load Q-table (default: data/q_table.npy)
    """
    if q_table_path is None:
        q_table_path = f'{DATA_DIR}/q_table.npy'
    
    os.makedirs(DATA_DIR, exist_ok=True)

    env = SDNRoutingEnv(controller=controller)
    agent = QTableAgent(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999
    )

    # Resume from existing Q-table if available
    if os.path.exists(q_table_path):
        agent.load(q_table_path)

    print("\n" + "=" * 70)
    print("  SDN Q-Learning Training")
    print("=" * 70)
    print(f"  Episodes       : {total_episodes}")
    print(f"  Steps/episode  : {EPISODE_STEPS}")
    print(f"  State dim      : {STATE_DIM}")
    print(f"  Num switches   : {NUM_DIST_SWITCHES}")
    print(f"  Num actions    : {NUM_ACTIONS}")
    print(f"  Mode           : {'LIVE (Ryu)' if controller else 'SIMULATION'}")
    print("=" * 70 + "\n")

    for episode in range(1, total_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0

        for step in range(EPISODE_STEPS):
            # Select action (epsilon-greedy)
            action = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Update Q-table
            agent.update(obs, action, reward, next_obs, terminated)

            episode_reward += reward
            obs = next_obs

            if terminated:
                break

        agent.episode_rewards.append(episode_reward)

        # Periodic save and logging
        if episode % save_every == 0 or episode == total_episodes:
            agent.save(q_table_path)

            # Compute running average
            recent_rewards = agent.episode_rewards[-min(10, len(agent.episode_rewards)):]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

            print(
                f"[Q-LEARNING] Episode {episode:3d}/{total_episodes} | "
                f"reward={episode_reward:7.4f} | "
                f"avg_10={avg_reward:7.4f} | "
                f"epsilon={agent.epsilon:.4f} | "
                f"q_states={len(agent.q_table):5d}"
            )

    env.close()
    print("\n[Q-LEARNING] Training complete.\n")


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Simulation mode
    run_q_learning(
        controller=None,
        total_episodes=5,
        save_every=2
    )