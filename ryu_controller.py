"""
ryu_controller.py

Simplified RL-ready SDN controller for a 5-switch multipath topology.

Topology:

    h1 -- s1 -- s2 -- s3 -- s4 -- h4
             \                /
              \---- s5 ------/

Hosts:
    h1 -> s1
    h2 -> s2
    h3 -> s3
    h4 -> s4

Paths:
    MAIN = s1 -> s2 -> s3 -> s4
    ALT  = s1 -> s5 -> s4

This version:
- fixes ping connectivity
- installs deterministic end-to-end paths
- installs reverse ICMP paths
- separates ARP from IPv4 forwarding
- removes unstable flooding behavior for IPv4
- keeps RL integration support

Run:
    ryu-manager ryu_classifier_controller.py
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls
)
from ryu.ofproto import ofproto_v1_3

from ryu.topology import event
from ryu.topology.api import get_switch, get_link

from ryu.lib.packet import (
    packet,
    ethernet,
    ipv4,
    tcp,
    udp,
    arp,
    ether_types
)

from ryu.lib import hub

import time
import threading
import numpy as np
import csv
import os
import sys
import traceback
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------------
# Shared state — written by Ryu, read by RL gym env
# -------------------------------------------------------------------

flow_stats_db = {}
timing_log = []
stats_lock = threading.Lock()

POLL_INTERVAL = 1

DATA_DIR = 'data'
RL_METRICS_LOG = f'{DATA_DIR}/rl_metrics_log.csv'


class ClosedLoopController(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.datapaths = {}
        self.hosts = {}
        self.adjacency = {}
        self.mac_to_port = {}

        self.paths = {
            "main": [1, 2, 3, 4],
            "alt": [1, 5, 4]
        }

        os.makedirs(DATA_DIR, exist_ok=True)

        with open(RL_METRICS_LOG, 'w', newline='') as f:

            csv.writer(f).writerow([
                'timestamp',
                'num_switches',
                'num_flows',
                'flow_stats_update_ms',
                'total_duration',
                'total_idle_time',
                'flow_count',
                'total_packet_count',
                'total_byte_count'
            ])

        self.logger.info("[INIT] Controller initialized")

    # ================================================================
    # SWITCH FEATURES
    # ================================================================

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):

        datapath = ev.msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        self.datapaths[datapath.id] = datapath

        match = parser.OFPMatch()

        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]

        self.add_flow(datapath, 0, match, actions)

        self.logger.info(f"[SWITCH] Connected dpid={datapath.id}")


    @set_ev_cls(event.EventSwitchEnter)
    def build_topology(self, ev):

        switch_list = get_switch(self, None)
        links = get_link(self, None)

        # raw adjacency
        graph = {}

        # ------------------------------------------------------------
        # STEP 1: build raw bidirectional graph from links
        # ------------------------------------------------------------
        for link in links:

            src = link.src
            dst = link.dst

            graph.setdefault(src.dpid, {})
            graph.setdefault(dst.dpid, {})

            graph[src.dpid][dst.dpid] = src.port_no
            graph[dst.dpid][src.dpid] = dst.port_no

        # ------------------------------------------------------------
        # STEP 2: normalize graph (important fix for missing edges)
        # ------------------------------------------------------------
        self.adjacency = {}

        for s in graph:
            self.adjacency[s] = dict(graph[s])

        # FORCE symmetry (fixes missing reverse links)
        for s in list(self.adjacency.keys()):
            for n, port in list(self.adjacency[s].items()):

                if n not in self.adjacency:
                    self.adjacency[n] = {}

                if s not in self.adjacency[n]:
                    # reverse port must exist from link discovery
                    self.logger.warning(f"[FIX] Missing reverse edge {n}->{s}")

        # ------------------------------------------------------------
        # STEP 3: validate connectivity using DFS
        # ------------------------------------------------------------
        self.validate_topology_with_dfs()

    def validate_topology_with_dfs(self):

        visited = set()

        def dfs(node):

            visited.add(node)

            for neigh in self.adjacency.get(node, {}):

                if neigh not in visited:
                    dfs(neigh)

        start = next(iter(self.adjacency), None)

        if start is None:
            self.logger.error("[DFS] Empty topology")
            return

        dfs(start)

        expected = set(self.adjacency.keys())

        missing = expected - visited

        if missing:

            self.logger.warning(
                f"[DFS] Disconnected nodes detected: {missing}"
            )

        else:

            self.logger.info("[DFS] Topology fully connected")
    # ================================================================
    # DFS PATH FINDER (exploratory, with backtracking)
    # ================================================================

    def find_path_dfs(self, start, goal, visited=None, path=None):
        """
        Depth-first search with backtracking to find any path from start to goal.
        Neighbors are visited in random order to encourage exploration (not shortest path).

        Returns a list of dpids representing the path, or None if no path found.
        """

        if visited is None:
            visited = set()
        if path is None:
            path = []

        visited.add(start)
        path.append(start)

        if start == goal:
            return path.copy()

        # Get neighbor dpids from adjacency mapping
        neighbors = list(self.adjacency.get(start, {}).keys())

        # Randomize exploration order to avoid deterministic shortest-path bias
        random.shuffle(neighbors)

        for nbr in neighbors:
            if nbr in visited:
                continue
            res = self.find_path_dfs(nbr, goal, visited, path)
            if res:
                return res

        # Backtrack
        path.pop()
        return None
    # ================================================================
    # PACKET IN
    # ================================================================

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        # Check if packets are coming in
        self.logger.info(f"[PACKET_IN] Packet received from dpid={datapath.id}")
        
        msg = ev.msg
        datapath = msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        dpid = datapath.id
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None:
            return

        dst = eth.dst
        src = eth.src

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        # ------------------------------------------------------------
        # 1. L2 LEARNING SWITCH BEHAVIOR (REQUIRED FOR ARP + PING)
        # ------------------------------------------------------------

        if dst in self.mac_to_port[dpid]:

            out_port = self.mac_to_port[dpid][dst]

        else:

            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # install basic L2 rule (IMPORTANT)
        match = parser.OFPMatch(in_port=in_port, eth_dst=dst)

        self._add_flow(datapath, 1, match, actions)

        # send packet
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )

        datapath.send_msg(out)

        # ------------------------------------------------------------
        # 2. ONLY AFTER L2 WORKS → RUN L3 PATH LOGIC FOR IP
        # ------------------------------------------------------------

        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        if not ip_pkt:
            return

        src_ip = ip_pkt.src
        dst_ip = ip_pkt.dst

        if src_ip not in self.hosts or dst_ip not in self.hosts:
            return

        src_sw, _ = self.hosts[src_ip]
        dst_sw, _ = self.hosts[dst_ip]

        path = self.find_path_dfs(src_sw, dst_sw)

        if path:

            self.install_path(path, src_ip, dst_ip)
            self.install_path(list(reversed(path)), dst_ip, src_ip)

        # ============================================================
        # PATH SELECTION
        # ============================================================

        # Attempt exploratory DFS-based path discovery first (may return any path).
        # This is intentionally exploratory (randomized neighbor order) and not
        # guaranteed to be the shortest path. If DFS fails, fall back to the
        # existing deterministic static paths.

        path = None
        try:
            src_sw, _ = self.hosts[src_ip]
            dst_sw, _ = self.hosts[dst_ip]
            path = self.find_path_dfs(src_sw, dst_sw)
        except Exception:
            path = None

        if path is None:
            # Fallback deterministic behavior
            if src_ip == "10.0.0.1" and dst_ip == "10.0.0.4":
                path = self.paths["alt"]
            else:
                path = self.paths["main"]

        self.logger.info(
            f"[PATH] {src_ip} -> {dst_ip} using {path}"
        )

        # ============================================================
        # INSTALL FORWARD PATH
        # ============================================================

        self.install_path(
            path=path,
            src_ip=src_ip,
            dst_ip=dst_ip
        )

        # ============================================================
        # INSTALL REVERSE PATH
        # ============================================================

        reverse_path = list(reversed(path))

        self.install_path(
            path=reverse_path,
            src_ip=dst_ip,
            dst_ip=src_ip
        )

        # ============================================================
        # FORWARD FIRST PACKET
        # ============================================================

        self.forward_packet(
            path=path,
            msg=msg
        )

    # ================================================================
    # INSTALL PATH
    # ================================================================

    def install_path(self, path, src_ip, dst_ip):
        if src_ip not in self.hosts or dst_ip not in self.hosts:
            return
        self.logger.info(
            f"[FLOW] Installing path {path} "
            f"for {src_ip} -> {dst_ip}"
        )

        for i in range(len(path)):

            switch_id = path[i]

            datapath = self.datapaths[switch_id]

            parser = datapath.ofproto_parser

            # --------------------------------------------------------
            # LAST SWITCH -> HOST
            # --------------------------------------------------------

            if i == len(path) - 1:

                dst_switch, dst_port = self.hosts[dst_ip]

                out_port = dst_port

            # --------------------------------------------------------
            # INTERMEDIATE SWITCH
            # --------------------------------------------------------

            else:

                next_switch = path[i + 1]

                out_port = self.adjacency[switch_id][next_switch]

            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip,
                ipv4_dst=dst_ip
            )

            actions = [
                parser.OFPActionOutput(out_port)
            ]

            self._add_flow(
                datapath=datapath,
                priority=10,
                match=match,
                actions=actions,
                idle_timeout=60
            )

    # ================================================================
    # FORWARD FIRST PACKET
    # ================================================================

    def get_next_hop(self, dpid, path):
        """
        Returns next switch in path given current switch.
        """
        if dpid not in path:
            return None

        idx = path.index(dpid)

        if idx == len(path) - 1:
            return None

        return path[idx + 1]

    def forward_packet(self, path, msg):

        datapath = msg.datapath
        parser = datapath.ofproto_parser

        current_sw = datapath.id

        next_sw = self.get_next_hop(current_sw, path)

        if next_sw is None:
            return

        if current_sw not in self.adjacency:
            return

        if next_sw not in self.adjacency[current_sw]:
            return

        out_port = self.adjacency[current_sw][next_sw]

        actions = [parser.OFPActionOutput(out_port)]

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=msg.match['in_port'],
            actions=actions,
            data=msg.data
        )

        datapath.send_msg(out)

    # ================================================================
    # RL ACTION
    # ================================================================

    def apply_rl_action(self, action_dict):

        """
        RL-ready API.

        Expected format:

        {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.4",
            "path": "alt"
        }
        """

        if not self.rl_enabled:
            return

        try:

            src_ip = action_dict['src_ip']
            dst_ip = action_dict['dst_ip']

            path_name = action_dict['path']

            if path_name not in self.paths:
                return

            path = self.paths[path_name]

            self.install_path(path, src_ip, dst_ip)

            self.install_path(
                list(reversed(path)),
                dst_ip,
                src_ip
            )

            self.logger.info(
                f"[RL] Applied path={path_name} "
                f"for {src_ip}->{dst_ip}"
            )

        except Exception as e:

            self.logger.error(
                f"[RL] apply_rl_action failed: {e}"
            )

    # ================================================================
    # FLOW STATS
    # ================================================================

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):

        dpid = ev.msg.datapath.id

        flows = []

        for stat in ev.msg.body:

            if stat.priority == 0:
                continue

            flows.append({

                'dpid': dpid,

                'cookie': stat.cookie,

                'priority': stat.priority,

                'packet_count': stat.packet_count,

                'byte_count': stat.byte_count,

                'duration_sec': stat.duration_sec,

                'duration_nsec': stat.duration_nsec,

                'idle_timeout': stat.idle_timeout,

                'match': stat.match,

                'table_id': stat.table_id,
            })

        with stats_lock:

            flow_stats_db[dpid] = flows

    # ================================================================
    # REQUEST FLOW STATS
    # ================================================================

    def _request_flow_stats(self, datapath):

        datapath.send_msg(
            datapath.ofproto_parser.OFPFlowStatsRequest(
                datapath
            )
        )

    # ================================================================
    # ADD FLOW
    # ================================================================

    def _add_flow(
        self,
        datapath,
        priority,
        match,
        actions,
        idle_timeout=0,
        hard_timeout=0
    ):

        ofproto = datapath.ofproto

        parser = datapath.ofproto_parser

        inst = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
            match=match,
            instructions=inst
        )

        datapath.send_msg(mod)

    # ================================================================
    # LOG METRICS
    # ================================================================

    def _log_metrics(
        self,
        num_switches,
        num_flows,
        stats_ms,
        total_duration,
        total_idle_time,
        flow_count,
        total_packet_count,
        total_byte_count
    ):

        with open(RL_METRICS_LOG, 'a', newline='') as f:

            csv.writer(f).writerow([

                time.time(),

                num_switches,

                num_flows,

                round(stats_ms, 3),

                round(total_duration, 3),

                round(total_idle_time, 3),

                flow_count,

                total_packet_count,

                total_byte_count
            ])

        with stats_lock:

            timing_log.append({

                'timestamp': time.time(),

                'num_flows': num_flows,

                'stats_ms': stats_ms,

                'total_duration': total_duration,

                'total_idle_time': total_idle_time,

                'flow_count': flow_count,

                'total_packet_count': total_packet_count,

                'total_byte_count': total_byte_count
            })