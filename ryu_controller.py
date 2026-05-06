"""
ryu_classifier_controller.py  (pure RL agent)

Single file passed to ryu-manager. Collects flow statistics and
spawns the RL training thread in the same process so they share
flow_stats_db directly without any sockets or IPC needed.

No supervised classifier — pure reinforcement learning agent learns
QoS policies directly from raw flow statistics and reward signals.

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

        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        with open(RL_METRICS_LOG, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'num_switches', 'num_flows',
                'flow_stats_update_ms'
            ])

        self.monitor_thread = hub.spawn(self._monitor_loop)

        self.rl_thread = hub.spawn(self._launch_rl_agent)

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
        
        self.logger.info(f"[PACKET_IN] dpid={datapath.id} in_port={in_port} buffer_id={msg.buffer_id}")
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            self.logger.debug("[PACKET_IN] No Ethernet protocol found")
            return
        dst  = eth.dst
        src  = eth.src
        dpid = datapath.id
        
        self.logger.info(f"[PACKET_IN] src={src} dst={dst}")
        
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = (self.mac_to_port[dpid][dst]
                    if dst in self.mac_to_port[dpid]
                    else ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self._add_flow(datapath, 10, match, actions)
            self.logger.info(f"[PACKET_IN] Installed flow for {dst} out on port {out_port}")
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out  = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
        self.logger.info(f"[PACKET_IN] Sent PacketOut to port {out_port}")

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dpid  = ev.msg.datapath.id
        flows = []

        self.logger.info(f"[STATS] dpid={dpid} total_entries={len(ev.msg.body)}")
        
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
            features_list, _ = self._extract_features()
            t2 = time.time()
            stats_ms = (t2 - t1) * 1000

            self._log_metrics(len(datapaths), len(features_list), stats_ms)
            
            if first_stats or len(features_list) > 0:
                self.logger.info(
                    f"[MONITOR] switches={len(datapaths)} | flows={len(features_list)} | "
                    f"stats_update={stats_ms:.1f}ms"
                )
                first_stats = False
            
            hub.sleep(max(0, POLL_INTERVAL - (time.time() - t0)))

    def _extract_features(self):
        features_list, flow_keys = [], []
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
        return features_list, flow_keys



    def _launch_rl_agent(self):
        self.logger.info("[RL] Waiting for switches to connect...")
        # Wait until at least 2 switches connect or 60s timeout
        for _ in range(60):
            hub.sleep(1)
            if len(self.datapaths) >= 2:
                self.logger.info(f"[RL] {len(self.datapaths)} switches connected, starting RL...")
                break
        else:
            self.logger.warning("[RL] Timeout waiting for switches, starting anyway...")

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
        if dpid not in self.datapaths:
            return
        datapath = self.datapaths[dpid]
        parser   = datapath.ofproto_parser
        ofproto  = datapath.ofproto
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

    def _log_metrics(self, num_switches, num_flows, stats_ms):
        with open(RL_METRICS_LOG, 'a', newline='') as f:
            csv.writer(f).writerow([
                time.time(), num_switches, num_flows,
                round(stats_ms, 3)
            ])
        with stats_lock:
            timing_log.append({
                'timestamp': time.time(), 'num_flows': num_flows,
                'stats_ms': stats_ms
            })
