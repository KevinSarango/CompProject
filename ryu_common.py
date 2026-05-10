import csv
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, arp, ipv4, tcp, icmp
from ryu.lib.packet import ether_types, in_proto


class BaseMultipathController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    POLICY_NAME = "BASE"
    METRICS_FILE = "data/base_metrics.csv"

    def __init__(self, *args, **kwargs):
        super(BaseMultipathController, self).__init__(*args, **kwargs)

        self.datapaths = {}
        self.flow_counter = 0

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

        os.makedirs("data", exist_ok=True)
        self.init_metrics_file()

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

    def choose_path(self, src, dst):
        raise NotImplementedError("Subclasses must implement choose_path")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp

        ofp = dp.ofproto
        parser = dp.ofproto_parser

        # Table miss only.
        # No more pre-installing all host-pair paths.
        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofp.OFPP_CONTROLLER,
                ofp.OFPCML_NO_BUFFER,
            )
        ]

        self.add_flow(dp, 0, match, actions)

        self.logger.info("[%s] Switch connected: s%s", self.POLICY_NAME, dp.id)

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
        """
        Install exact 5-tuple TCP rules for one TrafPy/iperf flow.
        This is the important fix: every unique iperf port can get its own path.
        """

        parser1 = self.datapaths[1].ofproto_parser

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
        """
        ICMP rule for ping/pingall.
        Lower priority than TCP so it does not override per-flow iperf decisions.
        """

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

        # Same-side traffic on s1.
        if src_side == "top" and dst_side == "top":
            parser = self.datapaths[1].ofproto_parser
            match = parser.OFPMatch(**match_fields)
            self.install_match(1, match, self.top_hosts[dst_ip], priority)
            return

        # Same-side traffic on s4.
        if src_side == "bottom" and dst_side == "bottom":
            parser = self.datapaths[4].ofproto_parser
            match = parser.OFPMatch(**match_fields)
            self.install_match(4, match, self.bottom_hosts[dst_ip], priority)
            return

        # Top -> Bottom
        if src_side == "top" and dst_side == "bottom":
            if path == "upper":
                # s1 -> s2 -> s4
                parser = self.datapaths[1].ofproto_parser
                self.install_match(1, parser.OFPMatch(**match_fields), 5, priority)

                parser = self.datapaths[2].ofproto_parser
                self.install_match(2, parser.OFPMatch(**match_fields), 2, priority)

                parser = self.datapaths[4].ofproto_parser
                self.install_match(4, parser.OFPMatch(**match_fields), self.bottom_hosts[dst_ip], priority)
            else:
                # s1 -> s3 -> s4
                parser = self.datapaths[1].ofproto_parser
                self.install_match(1, parser.OFPMatch(**match_fields), 6, priority)

                parser = self.datapaths[3].ofproto_parser
                self.install_match(3, parser.OFPMatch(**match_fields), 2, priority)

                parser = self.datapaths[4].ofproto_parser
                self.install_match(4, parser.OFPMatch(**match_fields), self.bottom_hosts[dst_ip], priority)

            return

        # Bottom -> Top
        if src_side == "bottom" and dst_side == "top":
            if path == "upper":
                # s4 -> s2 -> s1
                parser = self.datapaths[4].ofproto_parser
                self.install_match(4, parser.OFPMatch(**match_fields), 5, priority)

                parser = self.datapaths[2].ofproto_parser
                self.install_match(2, parser.OFPMatch(**match_fields), 1, priority)

                parser = self.datapaths[1].ofproto_parser
                self.install_match(1, parser.OFPMatch(**match_fields), self.top_hosts[dst_ip], priority)
            else:
                # s4 -> s3 -> s1
                parser = self.datapaths[4].ofproto_parser
                self.install_match(4, parser.OFPMatch(**match_fields), 6, priority)

                parser = self.datapaths[3].ofproto_parser
                self.install_match(3, parser.OFPMatch(**match_fields), 1, priority)

                parser = self.datapaths[1].ofproto_parser
                self.install_match(1, parser.OFPMatch(**match_fields), self.top_hosts[dst_ip], priority)

            return

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath

        pkt = packet.Packet(msg.data)

        eth_pkt = pkt.get_protocol(ethernet.ethernet)

        if eth_pkt is None:
            return

        # Ignore LLDP and other non-IP/ARP control traffic.
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

        # TCP / iperf flow.
        tcp_pkt = pkt.get_protocol(tcp.tcp)

        if tcp_pkt is not None:
            path = self.choose_path(src_ip, dst_ip)

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

        # ICMP / ping.
        icmp_pkt = pkt.get_protocol(icmp.icmp)

        if icmp_pkt is not None:
            path = self.choose_path(src_ip, dst_ip)

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
        """
        Sends the triggering packet back through the switch after rules are installed.
        The next packets in the flow should match installed rules.
        """

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
