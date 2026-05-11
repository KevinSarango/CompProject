import csv
import os
import time

from config import PATHS, TOPO_MODE
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import arp, ethernet, ether_types, icmp, in_proto, ipv4, packet, tcp
from ryu.ofproto import ofproto_v1_3


class BaseMultipathController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    POLICY_NAME = "BASE"
    METRICS_FILE = "data/base_metrics.csv"

    def __init__(self, *args, **kwargs):
        super(BaseMultipathController, self).__init__(*args, **kwargs)

        self.datapaths = {}
        self.flow_counter = 0

        # Diamond:
        #   s1 = top aggregation
        #   s4 = bottom aggregation
        #
        # Three-path:
        #   s1 = top aggregation
        #   s5 = bottom aggregation
        self.top_dpid = 1
        self.bottom_dpid = 4 if TOPO_MODE == "diamond" else 5

        self.ip_to_mac = {
            "10.0.0.1": "00:00:00:00:00:01",
            "10.0.0.2": "00:00:00:00:00:02",
            "10.0.0.3": "00:00:00:00:00:03",
            "10.0.0.4": "00:00:00:00:00:04",
            "10.0.0.5": "00:00:00:00:00:05",
            "10.0.0.6": "00:00:00:00:00:06",
            "10.0.0.7": "00:00:00:00:00:07",
            "10.0.0.8": "00:00:00:00:00:08",
        }

        self.top_hosts = {
            "10.0.0.1": 1,
            "10.0.0.2": 2,
            "10.0.0.3": 3,
            "10.0.0.4": 4,
        }

        self.bottom_hosts = {
            "10.0.0.5": 1,
            "10.0.0.6": 2,
            "10.0.0.7": 3,
            "10.0.0.8": 4,
        }

        self.path_ports = self.build_path_ports()

        os.makedirs("data", exist_ok=True)
        self.init_metrics_file()

    def build_path_ports(self):
        if TOPO_MODE == "diamond":
            return {
                "upper": {
                    "middle_dpid": 2,
                    "s1_to_middle": 5,
                    "middle_to_s1": 1,
                    "middle_to_bottom": 2,
                    "bottom_to_middle": 5,
                },
                "lower": {
                    "middle_dpid": 3,
                    "s1_to_middle": 6,
                    "middle_to_s1": 1,
                    "middle_to_bottom": 2,
                    "bottom_to_middle": 6,
                },
            }

        # three_path topology:
        #   low_delay: s1 -> s2 -> s5
        #   balanced:  s1 -> s3 -> s5
        #   high_bw:   s1 -> s4 -> s5
        return {
            "low_delay": {
                "middle_dpid": 2,
                "s1_to_middle": 5,
                "middle_to_s1": 1,
                "middle_to_bottom": 2,
                "bottom_to_middle": 5,
            },
            "balanced": {
                "middle_dpid": 3,
                "s1_to_middle": 6,
                "middle_to_s1": 1,
                "middle_to_bottom": 2,
                "bottom_to_middle": 6,
            },
            "high_bw": {
                "middle_dpid": 4,
                "s1_to_middle": 7,
                "middle_to_s1": 1,
                "middle_to_bottom": 2,
                "bottom_to_middle": 7,
            },
        }

    def init_metrics_file(self):
        with open(self.METRICS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "policy",
                "proto",
                "src",
                "dst",
                "src_port",
                "dst_port",
                "path",
            ])

    def log_decision(self, proto, src, dst, src_port, dst_port, path):
        with open(self.METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                self.POLICY_NAME,
                proto,
                src,
                dst,
                src_port,
                dst_port,
                path,
            ])

    def choose_path(self, src, dst, flow_info=None):
        raise NotImplementedError("Subclasses must implement choose_path")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp

        ofp = dp.ofproto
        parser = dp.ofproto_parser

        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofp.OFPP_CONTROLLER,
                ofp.OFPCML_NO_BUFFER,
            )
        ]

        self.add_flow(dp, 0, match, actions)

        self.logger.info(
            "[%s] Switch connected: s%s | TOPO_MODE=%s",
            self.POLICY_NAME,
            dp.id,
            TOPO_MODE,
        )

    def add_flow(self, dp, priority, match, actions, idle_timeout=30, hard_timeout=0):
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        inst = [
            parser.OFPInstructionActions(
                ofp.OFPIT_APPLY_ACTIONS,
                actions,
            )
        ]

        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )

        dp.send_msg(mod)

    def install_match(self, dpid, match, out_port, priority=200):
        dp = self.datapaths.get(dpid)

        if dp is None:
            self.logger.warning("Datapath s%s not connected yet", dpid)
            return

        parser = dp.ofproto_parser
        actions = [parser.OFPActionOutput(out_port)]
        self.add_flow(dp, priority, match, actions)

    def get_host_side(self, ip_addr):
        if ip_addr in self.top_hosts:
            return "top"

        if ip_addr in self.bottom_hosts:
            return "bottom"

        return None

    def install_tcp_path(self, src_ip, dst_ip, tcp_src, tcp_dst, path):
        forward_match = {
            "eth_type": ether_types.ETH_TYPE_IP,
            "ip_proto": in_proto.IPPROTO_TCP,
            "ipv4_src": src_ip,
            "ipv4_dst": dst_ip,
            "tcp_src": tcp_src,
            "tcp_dst": tcp_dst,
        }

        reverse_match = {
            "eth_type": ether_types.ETH_TYPE_IP,
            "ip_proto": in_proto.IPPROTO_TCP,
            "ipv4_src": dst_ip,
            "ipv4_dst": src_ip,
            "tcp_src": tcp_dst,
            "tcp_dst": tcp_src,
        }

        self.install_path_matches(forward_match, src_ip, dst_ip, path, priority=300)
        self.install_path_matches(reverse_match, dst_ip, src_ip, path, priority=300)

    def install_icmp_path(self, src_ip, dst_ip, path):
        forward_match = {
            "eth_type": ether_types.ETH_TYPE_IP,
            "ip_proto": in_proto.IPPROTO_ICMP,
            "ipv4_src": src_ip,
            "ipv4_dst": dst_ip,
        }

        reverse_match = {
            "eth_type": ether_types.ETH_TYPE_IP,
            "ip_proto": in_proto.IPPROTO_ICMP,
            "ipv4_src": dst_ip,
            "ipv4_dst": src_ip,
        }

        self.install_path_matches(forward_match, src_ip, dst_ip, path, priority=150)
        self.install_path_matches(reverse_match, dst_ip, src_ip, path, priority=150)

    def install_path_matches(self, match_fields, src_ip, dst_ip, path, priority):
        src_side = self.get_host_side(src_ip)
        dst_side = self.get_host_side(dst_ip)

        if src_side is None or dst_side is None:
            return

        # Same-side top traffic stays on s1.
        if src_side == "top" and dst_side == "top":
            parser = self.datapaths[self.top_dpid].ofproto_parser
            match = parser.OFPMatch(**match_fields)
            self.install_match(self.top_dpid, match, self.top_hosts[dst_ip], priority)
            return

        # Same-side bottom traffic stays on bottom aggregation switch.
        # Diamond bottom is s4. Three-path bottom is s5.
        if src_side == "bottom" and dst_side == "bottom":
            parser = self.datapaths[self.bottom_dpid].ofproto_parser
            match = parser.OFPMatch(**match_fields)
            self.install_match(
                self.bottom_dpid,
                match,
                self.bottom_hosts[dst_ip],
                priority,
            )
            return

        if path not in self.path_ports:
            path = PATHS[0]

        path_info = self.path_ports[path]
        middle_dpid = path_info["middle_dpid"]

        # Top -> Bottom
        if src_side == "top" and dst_side == "bottom":
            parser = self.datapaths[self.top_dpid].ofproto_parser
            self.install_match(
                self.top_dpid,
                parser.OFPMatch(**match_fields),
                path_info["s1_to_middle"],
                priority,
            )

            parser = self.datapaths[middle_dpid].ofproto_parser
            self.install_match(
                middle_dpid,
                parser.OFPMatch(**match_fields),
                path_info["middle_to_bottom"],
                priority,
            )

            parser = self.datapaths[self.bottom_dpid].ofproto_parser
            self.install_match(
                self.bottom_dpid,
                parser.OFPMatch(**match_fields),
                self.bottom_hosts[dst_ip],
                priority,
            )

            return

        # Bottom -> Top
        if src_side == "bottom" and dst_side == "top":
            parser = self.datapaths[self.bottom_dpid].ofproto_parser
            self.install_match(
                self.bottom_dpid,
                parser.OFPMatch(**match_fields),
                path_info["bottom_to_middle"],
                priority,
            )

            parser = self.datapaths[middle_dpid].ofproto_parser
            self.install_match(
                middle_dpid,
                parser.OFPMatch(**match_fields),
                path_info["middle_to_s1"],
                priority,
            )

            parser = self.datapaths[self.top_dpid].ofproto_parser
            self.install_match(
                self.top_dpid,
                parser.OFPMatch(**match_fields),
                self.top_hosts[dst_ip],
                priority,
            )

            return

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath

        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)

        if eth_pkt is None:
            return

        if eth_pkt.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        arp_pkt = pkt.get_protocol(arp.arp)

        if arp_pkt is not None:
            self.handle_arp(dp, msg, eth_pkt, arp_pkt)
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        if ip_pkt is None:
            return

        src_ip = ip_pkt.src
        dst_ip = ip_pkt.dst

        if src_ip not in self.ip_to_mac or dst_ip not in self.ip_to_mac:
            return

        tcp_pkt = pkt.get_protocol(tcp.tcp)

        if tcp_pkt is not None:
            flow_info = {
                "tcp_src": tcp_pkt.src_port,
                "tcp_dst": tcp_pkt.dst_port,
            }

            path = self.choose_path(src_ip, dst_ip, flow_info=flow_info)

            self.log_decision(
                proto="tcp",
                src=src_ip,
                dst=dst_ip,
                src_port=tcp_pkt.src_port,
                dst_port=tcp_pkt.dst_port,
                path=path,
            )

            self.install_tcp_path(
                src_ip=src_ip,
                dst_ip=dst_ip,
                tcp_src=tcp_pkt.src_port,
                tcp_dst=tcp_pkt.dst_port,
                path=path,
            )

            self.send_packet_out(dp, msg)
            return

        icmp_pkt = pkt.get_protocol(icmp.icmp)

        if icmp_pkt is not None:
            path = self.choose_path(src_ip, dst_ip, flow_info=None)

            self.log_decision(
                proto="icmp",
                src=src_ip,
                dst=dst_ip,
                src_port="",
                dst_port="",
                path=path,
            )

            self.install_icmp_path(src_ip, dst_ip, path)
            self.send_packet_out(dp, msg)
            return

    def handle_arp(self, dp, msg, eth_pkt, arp_pkt):
        if arp_pkt.opcode != arp.ARP_REQUEST:
            return

        target_ip = arp_pkt.dst_ip

        if target_ip not in self.ip_to_mac:
            return

        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        reply = packet.Packet()

        reply.add_protocol(
            ethernet.ethernet(
                dst=eth_pkt.src,
                src=self.ip_to_mac[target_ip],
                ethertype=ether_types.ETH_TYPE_ARP,
            )
        )

        reply.add_protocol(
            arp.arp(
                opcode=arp.ARP_REPLY,
                src_mac=self.ip_to_mac[target_ip],
                src_ip=target_ip,
                dst_mac=arp_pkt.src_mac,
                dst_ip=arp_pkt.src_ip,
            )
        )

        reply.serialize()

        actions = [parser.OFPActionOutput(in_port)]

        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=ofp.OFP_NO_BUFFER,
            in_port=ofp.OFPP_CONTROLLER,
            actions=actions,
            data=reply.data,
        )

        dp.send_msg(out)

    def send_packet_out(self, dp, msg):
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        actions = [parser.OFPActionOutput(ofp.OFPP_TABLE)]

        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None,
        )

        dp.send_msg(out)
