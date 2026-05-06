"""
sdn_gym_env.py

OpenAI Gym environment that wraps the Ryu SDN controller for
Q-learning based traffic management.

The RL agent observes raw flow statistics collected directly from
OFPFlowStatsReply messages and selects QoS queue actions for each
distribution switch — no ML classifier involved. The agent IS the
traffic manager, learning which queue policies work best based on
reward signals derived from live network performance.

State space  : Per-switch aggregated flow statistics (bytes/sec,
               packets/sec, avg packet size, active flows, protocol mix)
Action space : Queue priority assignment per distribution switch
               (0=default, 1=voip-priority, 2=video-priority,
                3=bulk-priority, 4=background-throttle)
Reward       : Throughput efficiency + flow fairness + low overhead

Dependencies:
    pip install gymnasium numpy

Usage:
    Imported by ryu_classifier_controller.py in _launch_rl_agent()
    Can also be tested standalone with:
        python3 sdn_gym_env.py
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
NUM_DIST_SWITCHES  = 2          # dist1..dist5 in campus topology
NUM_ACTIONS        = 5          # queue policy choices per switch
POLL_INTERVAL      = 1.0        # seconds between Ryu stats polls
OBS_HISTORY        = 3          # how many past intervals to stack in state
EPISODE_STEPS      = 20         # steps per training episode (~5min real time)

# Queue policy definitions — what each action ID means at the OFP level
QUEUE_POLICIES = {
    0: {'name': 'default',             'queues': {0: 100, 1: 100, 2: 100, 3: 100}},
    1: {'name': 'voip_priority',       'queues': {0: 5,   1: 10,  2: 20,  3: 65}},
    2: {'name': 'video_priority',      'queues': {0: 5,   1: 15,  2: 60,  3: 20}},
    3: {'name': 'bulk_priority',       'queues': {0: 5,   1: 60,  2: 20,  3: 15}},
    4: {'name': 'background_throttle', 'queues': {0: 60,  1: 15,  2: 15,  3: 10}},
}

# Features extracted per switch per polling interval
#   0: bytes_per_sec      (throughput)
#   1: packets_per_sec    (packet rate)
#   2: avg_packet_size    (bytes/packet — proxy for traffic type)
#   3: active_flows       (flow table pressure)
#   4: flow_duration_mean (average flow lifetime in seconds)
#   5: byte_rate_std      (variance in throughput — congestion signal)
#   6: pkt_rate_std       (variance in packet rate)
FEATURES_PER_SWITCH = 7
STATE_DIM = NUM_DIST_SWITCHES * FEATURES_PER_SWITCH * OBS_HISTORY


# ---------------------------------------------------------------------------
# Q-Table agent (tabular Q-learning with discretized state)
# ---------------------------------------------------------------------------

class QTableAgent:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.

    Because the raw state is continuous, we bin each feature
    into NUM_BINS buckets so the Q-table remains tractable.
    State key is a tuple of bin indices across all features.
    """

    NUM_BINS = 5

    # Rough expected ranges for each feature (for binning)
    FEATURE_RANGES = [
        (0,       100e6),   # bytes_per_sec      0 to 100 Mbps
        (0,       100e3),   # packets_per_sec    0 to 100k pps
        (0,       1500),    # avg_packet_size    0 to MTU
        (0,       50),      # active_flows       0 to 50
        (0,       300),     # flow_duration_mean 0 to 5min
        (0,       50e6),    # byte_rate_std
        (0,       50e3),    # pkt_rate_std
    ]

    def __init__(self,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.999):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: defaultdict so unseen states start at 0
        # Key  : tuple of discretized feature bins
        # Value: numpy array of shape (NUM_DIST_SWITCHES, NUM_ACTIONS)
        self.q_table = defaultdict(
            lambda: np.zeros((NUM_DIST_SWITCHES, NUM_ACTIONS))
        )
        self.episode_rewards  = []
        self.total_steps      = 0

    def _discretize(self, obs: np.ndarray) -> tuple:
        """
        Map continuous observation vector to a discrete tuple key.
        obs shape: (STATE_DIM,) = (NUM_SWITCHES * FEATURES * HISTORY,)
        We discretize only the most recent interval's features for the key.
        
        NOTE: obs is already normalized to [0, 1] by _normalise(), so we bin
              directly without applying FEATURE_RANGES again.
        """
        # Take only the last interval's features (already normalized to [0, 1])
        recent = obs[-NUM_DIST_SWITCHES * FEATURES_PER_SWITCH:]
        bins = []
        for sw in range(NUM_DIST_SWITCHES):
            for feat in range(FEATURES_PER_SWITCH):
                val   = recent[sw * FEATURES_PER_SWITCH + feat]
                # val is already in [0, 1], so bin directly
                # val=0.0 -> bin 0, val=0.5 -> bin 2.5->2, val=1.0 -> bin 4
                b     = int(np.clip(val, 0.0, 0.9999) * self.NUM_BINS)
                bins.append(b)
        return tuple(bins)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Epsilon-greedy action selection.
        Returns array of shape (NUM_DIST_SWITCHES,) with action per switch.
        """
        if random.random() < self.epsilon:
            # Explore: random action per switch
            return np.array([random.randint(0, NUM_ACTIONS - 1)
                             for _ in range(NUM_DIST_SWITCHES)])
        else:
            # Exploit: greedy action per switch
            state_key = self._discretize(obs)
            q_vals    = self.q_table[state_key]  # (NUM_SWITCHES, NUM_ACTIONS)
            return np.argmax(q_vals, axis=1)      # best action per switch

    def update(self, obs, actions, reward, next_obs, done):
        """
        Q-learning update:
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        Applied independently per switch (shared reward).
        """
        state_key      = self._discretize(obs)
        next_state_key = self._discretize(next_obs)

        q_current = self.q_table[state_key]
        q_next    = self.q_table[next_state_key]

        for sw_idx, action in enumerate(actions):
            best_next  = np.max(q_next[sw_idx])
            td_target  = reward + (0.0 if done else self.gamma * best_next)
            td_error   = td_target - q_current[sw_idx, action]
            q_current[sw_idx, action] += self.alpha * td_error

        # Decay exploration
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)
        self.total_steps += 1

    def save(self, path=None):
        """Persist Q-table and metadata."""
        if path is None:
            path = f'{DATA_DIR}/q_table.npy'
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            'q_table':  dict(self.q_table),
            'epsilon':  self.epsilon,
            'steps':    self.total_steps,
            'rewards':  self.episode_rewards,
        }
        np.save(path, data, allow_pickle=True)
        print(f"[Q-AGENT] Saved Q-table -> {path}  "
              f"({len(self.q_table)} unique states)")

    def load(self, path=None):
        """Restore Q-table from disk."""
        if path is None:
            path = f'{DATA_DIR}/q_table.npy'
        data = np.load(path, allow_pickle=True).item()
        for k, v in data['q_table'].items():
            self.q_table[k] = v
        self.epsilon       = data.get('epsilon', self.epsilon_min)
        self.total_steps   = data.get('steps', 0)
        self.episode_rewards = data.get('rewards', [])
        print(f"[Q-AGENT] Loaded Q-table from {path}  "
              f"({len(self.q_table)} states, "
              f"epsilon={self.epsilon:.3f})")


# ---------------------------------------------------------------------------
# Gym Environment
# ---------------------------------------------------------------------------

class SDNRoutingEnv(gym.Env):
    """
    OpenAI Gym environment for SDN Q-learning.

    Observation:
        Flattened vector of shape (STATE_DIM,) representing per-switch
        flow statistics stacked over the last OBS_HISTORY intervals.
        All values are normalised to [0, 1] before being returned.

    Action:
        MultiDiscrete([NUM_ACTIONS] * NUM_DIST_SWITCHES)
        Each element is the queue policy for that distribution switch.

    Reward:
        Composite signal computed from the live flow statistics snapshot:
          + throughput_score  : reward high aggregate bytes/sec
          + fairness_score    : penalise switches with no traffic (dead links)
          - overhead_penalty  : penalise large number of active flows
            (flow table bloat increases controller processing time)
          - congestion_penalty: penalise high variance in byte rates
            (sign of burst/congestion)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, controller=None):
        """
        Parameters
        ----------
        controller : ClosedLoopController | None
            Live Ryu controller instance.  When None the env runs in
            simulation mode (useful for unit tests and offline training).
        """
        super().__init__()

        self.controller  = controller
        self._sim_mode   = (controller is None)
        self._step_count = 0
        self._episode    = 0
        self._logged_dpids = False

        print(f"\n[GYM] Initializing SDNRoutingEnv")
        print(f"[GYM] Mode: {'SIMULATION' if self._sim_mode else 'LIVE (connected to Ryu)'}")
        print(f"[GYM] Controller: {controller}")
        print(f"[GYM] State dim: {STATE_DIM}")
        print(f"[GYM] Num dist switches: {NUM_DIST_SWITCHES}\n")

        # Observation: STATE_DIM normalised floats in [0,1]
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (STATE_DIM,),
            dtype = np.float32
        )

        # Action: one queue policy choice per distribution switch
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTIONS] * NUM_DIST_SWITCHES
        )

        # Ring buffer — last OBS_HISTORY raw feature matrices
        # Shape per entry: (NUM_DIST_SWITCHES, FEATURES_PER_SWITCH)
        self._obs_buffer = deque(
            [np.zeros((NUM_DIST_SWITCHES, FEATURES_PER_SWITCH),
                      dtype=np.float32)],
            maxlen=OBS_HISTORY
        )

        # Track previous byte counts for delta calculation
        self._prev_bytes   = defaultdict(float)
        self._prev_packets = defaultdict(float)
        self._prev_time    = time.time()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._episode   += 1
        self._obs_buffer = deque(
            [np.zeros((NUM_DIST_SWITCHES, FEATURES_PER_SWITCH),
                      dtype=np.float32)],
            maxlen=OBS_HISTORY
        )
        self._prev_bytes.clear()
        self._prev_packets.clear()
        self._prev_time = time.time()

        obs  = self._get_observation()
        info = {'episode': self._episode}
        return obs, info

    def step(self, action: np.ndarray):
        # Apply action
        self._apply_action(action)

        # Use hub.sleep if available, otherwise time.sleep
        try:
            from ryu.lib import hub
            hub.sleep(POLL_INTERVAL)
        except ImportError:
            time.sleep(POLL_INTERVAL)

        # Collect observation and reward
        obs    = self._get_observation()
        reward = self._compute_reward()

        self._step_count += 1
        terminated = (self._step_count >= EPISODE_STEPS)
        truncated  = False

        info = {
            'step':    self._step_count,
            'episode': self._episode,
            'action':  action.tolist(),
            'reward':  reward,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Print a human-readable summary of the current state."""
        raw = self._get_raw_stats()
        print("\n" + "=" * 60)
        print(f"  SDN Gym | Episode {self._episode} | Step {self._step_count}")
        print("=" * 60)
        for i, sw_stats in enumerate(raw):
            bps  = sw_stats[0] / 1e6   # to Mbps
            pps  = sw_stats[1] / 1e3   # to kpps
            aps  = sw_stats[2]          # bytes
            nf   = int(sw_stats[3])     # active flows
            dur  = sw_stats[4]          # seconds
            print(f"  dist{i+1}:  {bps:6.2f} Mbps  |  {pps:5.2f} kpps  |  "
                  f"{aps:4.0f} B/pkt  |  {nf:2d} flows  |  {dur:.1f}s avg dur")
        print("=" * 60)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw_stats(self) -> np.ndarray:
        """
        Pull flow statistics from the Ryu controller's shared
        flow_stats_db, then aggregate per distribution switch.

        Returns ndarray of shape (NUM_DIST_SWITCHES, FEATURES_PER_SWITCH).

        Distribution switch dpids are 9-13 in the campus topology:
            dpid  9 -> dist1 (academic)
            dpid 10 -> dist2 (dormitory)
            dpid 11 -> dist3 (admin)
            dpid 12 -> dist4 (datacenter)
            dpid 13 -> dist5 (library)

        If in simulation mode, synthetic stats are returned.
        """
        if self._sim_mode:
            return self._simulate_stats()

        # Import shared state from controller module
        try:
            import ryu_controller as ctrl_mod
            db   = ctrl_mod.flow_stats_db
            lock = ctrl_mod.stats_lock
        except ImportError:
            print("[GYM] WARNING: Could not import ryu_controller, using simulation")
            return self._simulate_stats()

        now     = time.time()
        dt      = max(now - self._prev_time, 1e-3)
        self._prev_time = now

        # dpids for dist1-dist5 (adjust if topology changes)
        DIST_DPIDS = [3, 4]

        result = np.zeros((NUM_DIST_SWITCHES, FEATURES_PER_SWITCH),
                          dtype=np.float32)

        with lock:
            snapshot = {d: list(f) for d, f in db.items()}

        # DEBUG: Log what switches we found
        if not hasattr(self, '_logged_dpids'):
            print(f"[GYM] Found dpids in controller: {list(snapshot.keys())}")
            print(f"[GYM] Looking for dpids: {DIST_DPIDS}")
            self._logged_dpids = True

        for sw_idx, dpid in enumerate(DIST_DPIDS):
            flows = snapshot.get(dpid, [])
            
            # DEBUG: Log flow counts
            if self._step_count % 10 == 0:
                print(f"[GYM] Step {self._step_count}: dpid={dpid} has {len(flows)} flows")
            
            if not flows:
                continue

            bytes_list   = []
            packets_list = []
            sizes        = []
            durations    = []
            byte_rates   = []
            pkt_rates    = []

            for flow in flows:
                bc  = flow.get('byte_count',   0)
                pc  = flow.get('packet_count',  0)
                dur = max(flow.get('duration_sec', 1), 1)

                key   = (dpid, flow.get('cookie', 0))
                delta_b = max(bc - self._prev_bytes.get(key, bc), 0)
                delta_p = max(pc - self._prev_packets.get(key, pc), 0)
                self._prev_bytes[key]   = bc
                self._prev_packets[key] = pc

                bps = delta_b / dt
                pps = delta_p / dt
                aps = (bc / pc) if pc > 0 else 0

                bytes_list.append(bps)
                packets_list.append(pps)
                sizes.append(aps)
                durations.append(dur)
                byte_rates.append(bps)
                pkt_rates.append(pps)

            n = len(flows)
            result[sw_idx] = [
                float(np.sum(bytes_list)),          # total bytes/sec
                float(np.sum(packets_list)),        # total packets/sec
                float(np.mean(sizes)) if sizes else 0.0,
                float(n),                           # active flow count
                float(np.mean(durations)) if durations else 0.0,
                float(np.std(byte_rates)) if len(byte_rates) > 1 else 0.0,
                float(np.std(pkt_rates))  if len(pkt_rates)  > 1 else 0.0,
            ]

        return result

    def _simulate_stats(self) -> np.ndarray:
        """
        Synthetic flow statistics for offline testing.
        Generates plausible values for each campus zone.
        """
        rng = np.random.default_rng()

        # Characteristic profiles per zone
        profiles = [
            # [bps_mean,  pps_mean, aps,  flows, dur]
            [5e6,   5000,  900,  4,  60],  # s3 — ingress switch
            [5e6,   5000,  900,  4,  60],  # s4 — egress switch
        ]

        result = np.zeros((NUM_DIST_SWITCHES, FEATURES_PER_SWITCH),
                          dtype=np.float32)
        for i, (bps, pps, aps, nf, dur) in enumerate(profiles):
            noise    = 0.15
            bps_s    = rng.normal(bps, bps * noise)
            pps_s    = rng.normal(pps, pps * noise)
            aps_s    = rng.normal(aps, aps * 0.05)
            dur_s    = rng.normal(dur, dur * 0.1)
            bps_std  = abs(rng.normal(bps * 0.1, bps * 0.05))
            pps_std  = abs(rng.normal(pps * 0.1, pps * 0.05))
            result[i] = [
                max(bps_s, 0), max(pps_s, 0), max(aps_s, 0),
                float(nf + rng.integers(-1, 2)),
                max(dur_s, 0), bps_std, pps_std,
            ]
        return result

    def _get_observation(self) -> np.ndarray:
        """
        Build normalised observation vector by:
          1. Fetching raw stats -> (NUM_SWITCHES, FEATURES)
          2. Normalising each feature to [0,1]
          3. Appending to history buffer
          4. Flattening the history buffer -> (STATE_DIM,)
        """
        raw  = self._get_raw_stats()
        norm = self._normalise(raw)
        self._obs_buffer.append(norm)

        # Pad if buffer not yet full
        history = list(self._obs_buffer)
        while len(history) < OBS_HISTORY:
            history.insert(0, np.zeros_like(norm))

        obs = np.concatenate(history, axis=None).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        """
        Normalise each feature column independently.
        Uses the same per-feature ranges as QTableAgent.FEATURE_RANGES.
        """
        ranges = np.array([
            [0,       100e6 ],   # bytes_per_sec
            [0,       100e3 ],   # packets_per_sec
            [0,       1500  ],   # avg_packet_size
            [0,       50    ],   # active_flows
            [0,       300   ],   # flow_duration_mean
            [0,       50e6  ],   # byte_rate_std
            [0,       50e3  ],   # pkt_rate_std
        ], dtype=np.float32)     # (FEATURES, 2)

        lo  = ranges[:, 0]       # (FEATURES,)
        hi  = ranges[:, 1]       # (FEATURES,)
        return np.clip((raw - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    def _compute_reward(self) -> float:
        """
        Composite reward responsive to queue policy changes.
        
        OLD approach (unresponsive):
          - Measured throughput (unaffected by queues)
          - Measured fairness (already perfect at 0.998)
        
        NEW approach (responsive to queue policies):
          - Flow completion efficiency: Ratio of completed flows to active flows
          - Duration smoothness: Penalize flows lasting very different times
          - Rate consistency: Prefer steady throughput over bursty
          - Overhead: Still penalize too many active flows
        
        This rewards queue policies that:
          1. Prevent packet loss (more flows complete successfully)
          2. Balance latency (flows finish in similar times)
          3. Maintain smooth traffic (low variance in throughput)
        """
        raw = self._get_raw_stats()
        
        # Extract features
        bps_per_sw      = raw[:, 0]      # bytes/sec
        pps_per_sw      = raw[:, 1]      # packets/sec
        avg_pkt_size    = raw[:, 2]      # bytes/packet (proxy for flow type)
        active_flows    = raw[:, 3]      # number of active flows
        flow_duration   = raw[:, 4]      # mean flow duration (seconds)
        bps_std         = raw[:, 5]      # variance in throughput
        pps_std         = raw[:, 6]      # variance in packet rate
        
        # ================================================================
        # Component 1: Flow Duration Efficiency
        # ================================================================
        # Good queue policies should complete flows faster
        # (lower duration = better priority scheduling)
        # Normalize to [0, 1]: 0 if flows last >60s, 1 if <1s
        eps = 1e-9
        duration_efficiency = 1.0 - np.clip(float(np.mean(flow_duration)) / 60.0, 0.0, 1.0)
        
        # ================================================================
        # Component 2: Throughput Consistency (lower variance = smoother)
        # ================================================================
        # Queue policies stabilize traffic by buffering
        # Low variance means good queue management
        total_bps = float(np.sum(bps_per_sw))
        mean_throughput_std = float(np.mean(bps_std))
        
        # Normalize std to [0, 1] for consistency score
        # High variance (congestion) = penalty
        consistency_score = np.exp(-mean_throughput_std / max(total_bps, eps))
        consistency_score = np.clip(consistency_score, 0.0, 1.0)
        
        # ================================================================
        # Component 3: Flow Load Balance (fairness)
        # ================================================================
        # Reward equal distribution across switches
        if np.sum(bps_per_sw) > eps:
            n = NUM_DIST_SWITCHES
            s = float(np.sum(bps_per_sw))
            sq = float(np.sum(bps_per_sw ** 2))
            fairness_score = (s ** 2) / (n * sq + eps)
        else:
            fairness_score = 0.0
        
        # ================================================================
        # Component 4: Overhead Penalty (too many flows = controller overhead)
        # ================================================================
        total_flows = float(np.sum(active_flows))
        FLOW_CEILING = 150.0
        overhead_penalty = min(total_flows / FLOW_CEILING, 1.0)
        
        # ================================================================
        # Component 5: Activity Penalty (flows should be active, not idle)
        # ================================================================
        # If avg packet size is very small OR very large, could indicate
        # protocol mismatch or stalled flows. Penalize extreme values.
        mean_pkt_size = float(np.mean(avg_pkt_size))
        activity_penalty = 0.0
        if mean_pkt_size < 100:  # Suspiciously small (idle packets?)
            activity_penalty += 0.1
        elif mean_pkt_size > 1400:  # Suspiciously large (fragmented?)
            activity_penalty += 0.05
        
        # ================================================================
        # Composite Reward
        # ================================================================
        reward = (
            0.35 * duration_efficiency    # Prioritize fast flow completion
            + 0.25 * consistency_score     # Prioritize smooth throughput
            + 0.20 * fairness_score        # Maintain fair load distribution
            - 0.15 * overhead_penalty      # Penalize controller overload
            - 0.05 * activity_penalty      # Penalize protocol anomalies
        )
        
        # DEBUG logging
        if self._step_count <= 5 or self._step_count % 20 == 0:
            print(f"[GYM-REWARD] Step {self._step_count}: "
                  f"duration_eff={duration_efficiency:.3f} "
                  f"consistency={consistency_score:.3f} "
                  f"fairness={fairness_score:.3f} "
                  f"overhead_pen={overhead_penalty:.3f} "
                  f"activity_pen={activity_penalty:.3f} "
                  f"→ reward={reward:.4f}")
        
        return float(np.clip(reward, -1.0, 1.0))

    def _apply_action(self, action: np.ndarray):
        """
        Translate discrete action indices into OFP queue rules and
        send them to the Ryu controller.

        Each action selects a QUEUE_POLICY for one distribution switch.
        The controller then applies OFPActionSetQueue for flows on that
        switch based on the policy's queue weights.

        In simulation mode this is a no-op.
        """
        if self._sim_mode:
            return   # Nothing to do in simulation

        if self.controller is None:
            return

        # dpids of switches — MUST MATCH YOUR TOPOLOGY!
        # For my_topology.py: s3=3, s4=4
        # For campus_topology.py: dist1=9, dist2=10, ..., dist5=13
        DIST_DPIDS = [3, 4]  # BOTTLENECK TOPOLOGY
        
        # DEBUG: Show what dpids controller has
        if self._step_count == 0:
            available = list(self.controller.datapaths.keys())
            print(f"\n[GYM ACTION] Available dpids: {available}")
            print(f"[GYM ACTION] Looking for dpids: {DIST_DPIDS}\n")

        for sw_idx, act in enumerate(action):
            if sw_idx >= len(DIST_DPIDS):
                break
            
            dpid   = DIST_DPIDS[sw_idx]
            policy = QUEUE_POLICIES[int(act)]

            if dpid not in self.controller.datapaths:
                if self._step_count % 10 == 0:
                    print(f"[GYM ACTION] Warning: dpid={dpid} not in controller.datapaths")
                continue

            datapath = self.controller.datapaths[dpid]
            parser   = datapath.ofproto_parser
            ofproto  = datapath.ofproto

            # Log action application
            if self._step_count <= 5 or self._step_count % 50 == 0:
                print(f"[GYM ACTION] Step {self._step_count}: dpid={dpid} "
                      f"policy={policy['name']} action={int(act)}")

            # Apply queue assignments for each priority level
            # by installing/refreshing flow rules per queue
            for queue_id, weight in policy['queues'].items():
                try:
                    match   = parser.OFPMatch()   # match-all for this switch
                    actions = [
                        parser.OFPActionSetQueue(int(queue_id)),
                        parser.OFPActionOutput(ofproto.OFPP_NORMAL)
                    ]
                    inst = [parser.OFPInstructionActions(
                        ofproto.OFPIT_APPLY_ACTIONS, actions)]
                    datapath.send_msg(parser.OFPFlowMod(
                        datapath=datapath,
                        cookie=0,
                        cookie_mask=0,
                        table_id=0,
                        command=ofproto.OFPFC_ADD,
                        idle_timeout=int(POLL_INTERVAL * 2),
                        hard_timeout=0,
                        priority=int(queue_id + 1),
                        buffer_id=ofproto.OFP_NO_BUFFER,
                        out_port=ofproto.OFPP_ANY,
                        out_group=ofproto.OFPG_ANY,
                        flags=0,
                        match=match,
                        instructions=inst
                    ))
                except Exception as e:
                    import traceback
                    print(f"[GYM ACTION] Error applying action: {e}")
                    print(traceback.format_exc())
                    print(f"[GYM] Action apply failed dpid={dpid} "
                          f"queue={queue_id}: {e}")
                    traceback.print_exc()

            print(f"[GYM] dist{sw_idx+1} (dpid={dpid}) -> "
                  f"policy='{policy['name']}'")


# ---------------------------------------------------------------------------
# Training loop — runs inside Ryu process or standalone
# ---------------------------------------------------------------------------

def run_q_learning(controller=None,
                   total_episodes=100,
                   save_every=25,
                   q_table_path=None):
    """
    Full Q-learning training loop with learning decay detection.

    Parameters
    ----------
    controller    : ClosedLoopController | None
    total_episodes: how many episodes to train
    save_every    : persist Q-table every N episodes
    q_table_path  : path to save/load Q-table (default: data/q_table.npy)
    """
    import os

    if q_table_path is None:
        q_table_path = f'{DATA_DIR}/q_table.npy'
    
    os.makedirs(DATA_DIR, exist_ok=True)

    env   = SDNRoutingEnv(controller=controller)
    agent = QTableAgent(
        alpha         = 0.1,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_min   = 0.05,
        epsilon_decay = 0.999,
    )

    # Resume from existing Q-table if available
    if os.path.exists(q_table_path):
        agent.load(q_table_path)

    # Initialize episode rewards log
    with open(f'{DATA_DIR}/rl_episode_rewards.log', 'w') as f:
        f.write('episode,reward,avg_10,avg_20,decay,num_states,epsilon\n')

    print("\n" + "=" * 60)
    print("  SDN Q-Learning Training")
    print(f"  Episodes       : {total_episodes}")
    print(f"  Steps/episode  : {EPISODE_STEPS}")
    print(f"  State dim      : {STATE_DIM}")
    print(f"  Actions/switch : {NUM_ACTIONS}")
    print(f"  Switches       : {NUM_DIST_SWITCHES}")
    print(f"  Mode           : {'LIVE (Ryu)' if controller else 'SIMULATION'}")
    print("=" * 60 + "\n")

    for episode in range(1, total_episodes + 1):
        obs, _          = env.reset()
        episode_reward  = 0.0
        episode_actions = []

        for step in range(EPISODE_STEPS):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, next_obs,
                         terminated or truncated)

            obs             = next_obs
            episode_reward += reward
            episode_actions.append(action.tolist())
            
            # DEBUG: Show first episode details
            if episode == 1 or episode % 25 == 0:
                raw_stats = env._get_raw_stats()
                state_key = agent._discretize(obs)
                if step == 0:
                    print(f"[EP {episode}] Raw stats: {raw_stats}")
                    print(f"[EP {episode}] Step {step} action={action} reward={reward:.4f} state_key={state_key[:7]}...")

            if terminated or truncated:
                break

        agent.episode_rewards.append(episode_reward)

        # Compute rolling averages
        avg_10 = np.mean(agent.episode_rewards[-10:]) if len(agent.episode_rewards) >= 10 else episode_reward
        avg_20 = np.mean(agent.episode_rewards[-20:]) if len(agent.episode_rewards) >= 20 else avg_10
        
        # Detect learning decay
        decay = avg_10 - avg_20
        decay_flag = "DECAY" if decay < -0.1 else ""

        # Logging
        print(f"[EP {episode:4d}/{total_episodes}] "
              f"reward={episode_reward:+6.3f}  "
              f"avg10={avg_10:+6.3f}  "
              f"epsilon={agent.epsilon:.3f}  "
              f"states={len(agent.q_table):4d}  "
              f"{decay_flag}")
        
        # Log to file for correlation analysis
        with open(f'{DATA_DIR}/rl_episode_rewards.log', 'a') as f:
            f.write(f'{episode},{episode_reward:.4f},{avg_10:.4f},{avg_20:.4f},'
                    f'{decay:.4f},{len(agent.q_table)},{agent.epsilon:.5f}\n')

        if episode % save_every == 0:
            agent.save(q_table_path)

    agent.save(q_table_path)
    env.close()
    print("\n[Q-LEARNING] Training complete.")
    print(f"[Q-LEARNING] Rewards logged to {DATA_DIR}/rl_episode_rewards.log")
    print(f"[Q-LEARNING] Correlate with {DATA_DIR}/traffic_scenario.log using analyze_correlation.py")
    return agent


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='SDN Q-Learning Gym — standalone simulation mode'
    )
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of training episodes (default: 10)')
    parser.add_argument('--steps', type=int, default=20,
                        help='Steps per episode (default: 20, ~poll intervals)')
    parser.add_argument('--no-sleep', action='store_true',
                        help='Skip POLL_INTERVAL sleep (fast sim mode)')
    args = parser.parse_args()

    # Monkey-patch sleep for fast offline testing
    if args.no_sleep:
        import sdn_gym_env as _self
        _self.POLL_INTERVAL = 0.0
        EPISODE_STEPS = args.steps

    print("[STANDALONE] Running in SIMULATION mode (no Ryu controller)")
    print("[STANDALONE] Use --no-sleep for fast offline training\n")

    agent = run_q_learning(
        controller     = None,
        total_episodes = args.episodes,
        save_every     = max(1, args.episodes // 4),
    )

    # Show a sample of learned Q-values
    print(f"\n[Q-TABLE] Learned {len(agent.q_table)} unique states")
    print("[Q-TABLE] Sample entries (state_key -> best_action per switch):")
    for i, (key, q_vals) in enumerate(list(agent.q_table.items())[:5]):
        best = np.argmax(q_vals, axis=1)
        policies = [QUEUE_POLICIES[a]['name'] for a in best]
        print(f"  State {i}: best_actions = {policies}")
