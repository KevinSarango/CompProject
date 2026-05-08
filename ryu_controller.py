"""
ryu_classifier_controller.py  (pure RL agent with multi-path routing)

Single file passed to ryu-manager. Collects flow statistics from all 6 switches
in the multi-path topology and spawns the RL training thread in the same process
so they share flow_stats_db directly without any sockets or IPC needed.

Multi-path topology (6 switches):
  - s1 (dpid=1): Decision point, routes to path1 or path2
  - s2, s3: Path 1 intermediate switches
  - s4 (dpid=4): Decision point, applies final QoS
  - s5, s6: Path 2 alternate route

Q table-based reinforcement learning agent learns
QoS policies and routing decisions directly from raw flow statistics
and reward signals that account for per-path congestion.

Run with:
    ryu-manager ryu_controller.py
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
from ryu.lib import hub

import time
import threading
import numpy as np
import csv
import os
import sys
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------------
# Shared state — written by Ryu, read by RL gym env
# -------------------------------------------------------------------
flow_stats_db = {}
timing_log    = []
stats_lock    = threading.Lock()

POLL_INTERVAL = 1
DATA_DIR = 'data'
RL_METRICS_LOG = f'{DATA_DIR}/rl_metrics_log.csv'


class ClosedLoopController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths    = {}
        self.mac_to_port  = {}
        self.flow_history = {}
        self.rl_enabled   = False  # Disable RL initially to allow MAC learning

        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # CSV header for metrics logging
        # Monitors all 6 switches in the multi-path topology: s1-s6
        with open(RL_METRICS_LOG, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'num_switches', 'num_flows',
                'flow_stats_update_ms', 'total_duration', 'total_idle_time',
                'flow_count', 'total_packet_count', 'total_byte_count'
            ])

        # self.monitor_thread = hub.spawn(self._monitor_loop)

        # self.rl_thread = hub.spawn(self._launch_rl_agent)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath
        self.logger.info(f"[SWITCH] Connected: dpid={datapath.id}")
        
        # Add a low-priority catch-all rule so controller still receives
        # unmatched traffic, but higher-priority RL policies can override it.
        match   = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)  # Low-priority fallback
        self.logger.info(f"[SWITCH] Installed low-priority catch-all on dpid={datapath.id} to send unmatched traffic to controller")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        in_port  = msg.match['in_port']
        
        # Minimal logging during learning phase to reduce spam
        # (debug level won't print by default)
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        dst  = eth.dst
        src  = eth.src
        dpid = datapath.id
        
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        # self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)
        
        out_port = (self.mac_to_port[dpid][dst]
                    if dst in self.mac_to_port[dpid]
                    else ofproto.OFPP_FLOOD)
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Only log when learning NEW MAC addresses (installing new flow)
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self._add_flow(datapath, 1, match, actions)

        # Send PacketOut (but don't log every one)
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out  = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dpid  = ev.msg.datapath.id
        flows = []

        self.logger.info(f"[STATS] dpid={dpid} total_entries={len(ev.msg.body)}")
        
        # DEBUG: Log first 5 entries before filtering
        for i, stat in enumerate(ev.msg.body[:5]):
            self.logger.info(
                f"  Entry {i}: priority={stat.priority} bytes={stat.byte_count} "
                f"packets={stat.packet_count} duration={stat.duration_sec}s"
            )
        
        for stat in ev.msg.body:
            if stat.priority == 0:
                continue
            flows.append({
                'dpid':          dpid,
                'cookie':        stat.cookie,
                'priority':      stat.priority,
                'packet_count':  stat.packet_count,
                'byte_count':    stat.byte_count,
                'duration_sec':  stat.duration_sec,
                'duration_nsec': stat.duration_nsec,
                'idle_timeout':  stat.idle_timeout,
                'match':         stat.match,
                'table_id':      stat.table_id,
            })
        with stats_lock:
            flow_stats_db[dpid] = flows

    def _monitor_loop(self):
        """Collect flow statistics and make available to RL agent."""
        first_stats = True
        while True:
            t0 = time.time()
            with stats_lock:
                datapaths = list(self.datapaths.values())
            
            if not datapaths:
                self.logger.debug("[MONITOR] Waiting for switches to connect...")
                hub.sleep(1)
                continue
                
            for dp in datapaths:
                self._request_flow_stats(dp)
            hub.sleep(0.1)

            t1 = time.time()
            features_list, _, total_duration, total_idle_time, flow_count, total_packet_count, total_byte_count = self._extract_features()
            t2 = time.time()
            stats_ms = (t2 - t1) * 1000

            self._log_metrics(len(datapaths), len(features_list), stats_ms, total_duration, total_idle_time, flow_count, total_packet_count, total_byte_count)
            
            if first_stats or len(features_list) > 0:
                self.logger.info(
                    f"[MONITOR] switches={len(datapaths)} | flows={len(features_list)} | "
                    f"stats_update={stats_ms:.1f}ms"
                )
                first_stats = False
            
            hub.sleep(max(0, POLL_INTERVAL - (time.time() - t0)))

    def _extract_features(self):
        features_list, flow_keys = [], []
        total_duration = 0.0
        total_idle_time = 0.0
        total_packet_count = 0
        total_byte_count = 0
        with stats_lock:
            snapshot = {d: list(f) for d, f in flow_stats_db.items()}
        for dpid, flows in snapshot.items():
            for flow in flows:
                key      = (dpid, flow['cookie'])
                prev     = self.flow_history.get(key, {
                    'byte_count': 0, 'packet_count': 0, 'duration_sec': 0})
                duration = max(flow['duration_sec'] + flow['duration_nsec'] / 1e9, 1e-6)
                delta_bytes   = max(flow['byte_count']   - prev['byte_count'],   0)
                delta_packets = max(flow['packet_count'] - prev['packet_count'], 0)
                self.flow_history[key] = {
                    'byte_count':   flow['byte_count'],
                    'packet_count': flow['packet_count'],
                    'duration_sec': flow['duration_sec'],
                }
                pkt_count  = flow['packet_count']
                byte_count = flow['byte_count']
                idle_time  = flow.get('idle_timeout', 0)
                features_list.append([
                    pkt_count, byte_count,
                    delta_packets / POLL_INTERVAL,
                    delta_bytes   / POLL_INTERVAL,
                    (byte_count / pkt_count) if pkt_count > 0 else 0,
                    duration,
                    byte_count / duration,
                    pkt_count  / duration,
                ])
                flow_keys.append((dpid, flow))
                total_duration += duration
                total_idle_time += idle_time
                total_packet_count += pkt_count
                total_byte_count += byte_count
        flow_count = len(features_list)
        return features_list, flow_keys, total_duration, total_idle_time, flow_count, total_packet_count, total_byte_count



    def _launch_rl_agent(self):
        self.logger.info("[RL] Waiting for switches to connect...")
        # Wait until all 6 switches connect (or at least 4) or 60s timeout
        target_switches = 6
        for _ in range(60):
            hub.sleep(1)
            if len(self.datapaths) >= target_switches:
                self.logger.info(f"[RL] All {len(self.datapaths)} switches connected.")
                break
            elif len(self.datapaths) >= 4:
                self.logger.info(f"[RL] {len(self.datapaths)} switches connected (target: {target_switches}), "
                               f"waiting a bit more...")
        else:
            self.logger.warning(f"[RL] Timeout waiting for {target_switches} switches "
                              f"({len(self.datapaths)} connected), proceeding...")

        # ====================================================================
        # MAC LEARNING PHASE: Allow basic learning bridge to populate MAC tables
        # This is CRITICAL before RL rules are installed
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("[RL] *** ENTERING MAC LEARNING PHASE (30 seconds) ***")
        self.logger.info("[RL] RL rules are DISABLED during this phase.")
        self.logger.info("[RL] Learning bridge will populate MAC tables automatically.")
        self.logger.info("[RL]")
        self.logger.info("[RL] IN MININET CLI, run these commands:")
        self.logger.info("[RL]   mininet> pingall")
        self.logger.info("[RL]   mininet> pingall")
        self.logger.info("[RL]")
        self.logger.info("[RL] Verify all hosts can ping each other before proceeding!")
        self.logger.info("="*70 + "\n")
        
        # Keep RL disabled during learning phase (default 30 seconds).
        # Prefer an explicit signal file created by the topology when connectivity
        # is verified. Fall back to a timeout if the file doesn't appear.
        learning_phase_duration = 30
        signal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'connectivity_verified.txt')
        self.logger.info(f"[RL] Waiting for connectivity signal file at {signal_path} (timeout {learning_phase_duration}s)...")
        waited = 0
        while waited < learning_phase_duration:
            if os.path.exists(signal_path):
                self.logger.info("[RL] Connectivity signal found — skipping remaining learning delay.")
                break
            hub.sleep(1)
            waited += 1
            remaining = learning_phase_duration - waited
            if remaining % 10 == 0 or remaining <= 5:
                self.logger.info(f"[RL] MAC learning phase: {remaining}s remaining...")
        else:
            self.logger.info("[RL] No connectivity signal found; proceeding after learning timeout.")
        
        # ====================================================================
        # ENABLE RL AGENT: Now start installing routing rules
        # ====================================================================
        self.rl_enabled = True
        self.logger.info("\n" + "="*70)
        self.logger.info("[RL] MAC LEARNING PHASE COMPLETE!")
        self.logger.info("[RL] *** ENABLING RL AGENT - INSTALLING ROUTING RULES ***")
        self.logger.info("="*70 + "\n")

        try:
            from sdn_gym_env import SDNRoutingEnv, run_q_learning
            self.logger.info("[RL] Starting pure Q-learning agent...")
            self.logger.info(f"[RL] Controller datapaths: {list(self.datapaths.keys())}")

            run_q_learning(
                controller     = self,
                total_episodes = 100,
                save_every     = 5,
                q_table_path   = f'{DATA_DIR}/q_table.npy'
            )
        except ImportError as e:
            self.logger.warning(f"[RL] Missing package: {e}")
            self.logger.warning(traceback.format_exc())
        except Exception as e:
            self.logger.error(f"[RL] Training error: {e}")
            self.logger.error(traceback.format_exc())

    def apply_rl_action(self, action_dict):
        dpid = action_dict['dpid']

        #Ignore RL actions until MAC learning phase is complete and rl_enabled is True
        if not getattr(self, 'rl_enabled', False):
            try:
                self.logger.debug(f"[RL] Ignoring action for dpid={dpid} because RL agent is not enabled yet.")
            except Exception:
                pass
            return
        

        if dpid not in self.datapaths:
            return
        datapath = self.datapaths[dpid]
        parser   = datapath.ofproto_parser
        ofproto  = datapath.ofproto
        self.logger.info(f"[RL] Applying action to dpid={dpid}: {action_dict}")
        try:
            match   = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=action_dict['src_ip'],
                ipv4_dst=action_dict['dst_ip']
            )
            actions = [
                parser.OFPActionSetQueue(action_dict['queue_id']),
                parser.OFPActionOutput(action_dict['out_port'])
            ]
            self._add_flow(datapath, action_dict['priority'],
                           match, actions, idle_timeout=30)
        except Exception as e:
            self.logger.warning(f"[RL] apply_rl_action failed: {e}")

    def _request_flow_stats(self, datapath):
        self.logger.debug(f"[STATS_REQUEST] Requesting stats from dpid={datapath.id}")
        datapath.send_msg(
            datapath.ofproto_parser.OFPFlowStatsRequest(datapath))

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        datapath.send_msg(parser.OFPFlowMod(
            datapath=datapath, priority=priority,
            idle_timeout=idle_timeout, hard_timeout=hard_timeout,
            match=match, instructions=inst))

    def _log_metrics(self, num_switches, num_flows, stats_ms, total_duration, total_idle_time, flow_count, total_packet_count, total_byte_count):
        with open(RL_METRICS_LOG, 'a', newline='') as f:
            csv.writer(f).writerow([
                time.time(), num_switches, num_flows,
                round(stats_ms, 3), round(total_duration, 3), round(total_idle_time, 3),
                flow_count, total_packet_count, total_byte_count
            ])
        with stats_lock:
            timing_log.append({
                'timestamp': time.time(), 'num_flows': num_flows,
                'stats_ms': stats_ms, 'total_duration': total_duration,
                'total_idle_time': total_idle_time, 'flow_count': flow_count,
                'total_packet_count': total_packet_count, 'total_byte_count': total_byte_count
            })
