import csv
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, arp, ipv4, tcp


from config import DEFAULT_FLOW_SIZE_KB


class BaseMultipathController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    POLICY_NAME = "BASE"
    METRICS_FILE = "data/base_metrics.csv"

    def __init__(self, *args, **kwargs):
        super(BaseMultipathController, self).__init__(*args, **kwargs)

        self.datapaths = {}
        self.installed = False
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

        self.port_to_size_kb = self.load_trafpy_flow_sizes()

        os.makedirs("data", exist_ok=True)
        self.init_metrics_file()

    def init_metrics_file(self):
        with open(self.METRICS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "policy",
                "src",
                "dst",
                "path",
                "proto",
                "tcp_dst",
                "flow_size_kb",
            ])

    def load_trafpy_flow_sizes(self):
        """
        automated_traffic_tests.py uses:
            port = 5001 + flow_id

        This lets Ryu map each TCP destination port back to the TrafPy flow size.
        """
        path = "data/trafpy_demands.csv"
        port_to_size = {}

        if not os.path.exists(path):
            return port_to_size

        with open(path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                flow_id = int(row["flow_id"])
                tcp_port = 5001 + flow_id
                port_to_size[tcp_port] = float(row["size_kb"])

        return port_to_size

    def log_decision(self, src, dst, path, proto="unknown", tcp_dst="", flow_size_kb=""):
        with open(self.METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                self.POLICY_NAME,
                src,
                dst,
                path,
                proto,
                tcp_dst,
                flow_size_kb,
            ])

    def choose_path(self, src, dst, tcp_dst=None, flow_size_kb=None):
        raise NotImplementedError("Subclasses must implement choose_path")

    def is_top_host(self, ip):
        return ip in self.top_hosts

    def is_bottom_host(self, ip):
        return ip in self.bottom_hosts

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp

        ofp = dp.ofproto
        parser = dp.ofproto_parser

        # Table-miss sends unknown traffic to controller.
        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofp.OFPP_CONTROLLER,
                ofp.OFPCML_NO_BUFFER,
            )
        ]

        self.add_flow(dp, 0, match, actions)

        self.logger.info("[%s] Switch connected: s%s", self.POLICY_NAME, dp.id)

        if len(self.datapaths) == 4 and not self.installed:
            self.install_static_connectivity()
            self.installed = True

    def add_flow(self, dp, priority, match, actions):
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
        )

        dp.send_msg(mod)

    def install_flow(self, dpid, priority, match_kwargs, out_port):
        dp = self.datapaths[dpid]
        parser = dp.ofproto_parser

        match = parser.OFPMatch(**match_kwargs)
        actions = [parser.OFPActionOutput(out_port)]

        self.add_flow(dp, priority, match, actions)

    def install_static_connectivity(self):
        """
        Installs:
        - same-side IPv4 rules
        - ICMP rules for pingall connectivity

        TCP top-to-bottom traffic is NOT preinstalled.
        TCP flows are handled dynamically per TrafPy/iperf flow.
        """

        hosts = list(self.top_hosts.keys()) + list(self.bottom_hosts.keys())

        for src in hosts:
            for dst in hosts:
                if src == dst:
                    continue

                src_top = self.is_top_host(src)
                dst_top = self.is_top_host(dst)

                # Same side on s1.
                if src_top and dst_top:
                    self.install_flow(
                        1,
                        100,
                        {
                            "eth_type": 0x0800,
                            "ipv4_src": src,
                            "ipv4_dst": dst,
                        },
                        self.top_hosts[dst],
                    )

                # Same side on s4.
                elif not src_top and not dst_top:
                    self.install_flow(
                        4,
                        100,
                        {
                            "eth_type": 0x0800,
                            "ipv4_src": src,
                            "ipv4_dst": dst,
                        },
                        self.bottom_hosts[dst],
                    )

                # Different sides: install ICMP only for pingall.
                else:
                    path = "upper" if int(dst.split(".")[-1]) % 2 == 0 else "lower"

                    self.install_ip_proto_path(
                        src=src,
                        dst=dst,
                        path=path,
                        ip_proto=1,
                        priority=90,
                    )

        self.logger.info(
            "[%s] Installed static same-side IPv4 and cross-side ICMP rules",
            self.POLICY_NAME,
        )

    def install_ip_proto_path(self, src, dst, path, ip_proto, priority=90):
        match_kwargs = {
            "eth_type": 0x0800,
            "ip_proto": ip_proto,
            "ipv4_src": src,
            "ipv4_dst": dst,
        }

        self.install_path_match(src, dst, path, match_kwargs, priority)

    def install_tcp_path(self, src, dst, path, tcp_dst):
        """
        Install TCP rules for one TrafPy/iperf flow.

        Forward flow:
            src -> dst, tcp_dst = iperf server port

        Reverse flow:
            dst -> src, tcp_src = iperf server port
        """

        forward_match = {
            "eth_type": 0x0800,
            "ip_proto": 6,
            "ipv4_src": src,
            "ipv4_dst": dst,
            "tcp_dst": tcp_dst,
        }

        reverse_match = {
            "eth_type": 0x0800,
            "ip_proto": 6,
            "ipv4_src": dst,
            "ipv4_dst": src,
            "tcp_src": tcp_dst,
        }

        self.install_path_match(src, dst, path, forward_match, priority=200)
        self.install_path_match(dst, src, path, reverse_match, priority=200)

    def install_path_match(self, src, dst, path, match_kwargs, priority):
        """
        Installs a match along either upper or lower path.
        Handles both top->bottom and bottom->top directions.
        """

        src_top = self.is_top_host(src)
        dst_top = self.is_top_host(dst)

        if src_top and not dst_top:
            # Top -> bottom
            if path == "upper":
                self.install_flow(1, priority, match_kwargs, 5)
                self.install_flow(2, priority, match_kwargs, 2)
                self.install_flow(4, priority, match_kwargs, self.bottom_hosts[dst])
            else:
                self.install_flow(1, priority, match_kwargs, 6)
                self.install_flow(3, priority, match_kwargs, 2)
                self.install_flow(4, priority, match_kwargs, self.bottom_hosts[dst])

        elif not src_top and dst_top:
            # Bottom -> top
            if path == "upper":
                self.install_flow(4, priority, match_kwargs, 5)
                self.install_flow(2, priority, match_kwargs, 1)
                self.install_flow(1, priority, match_kwargs, self.top_hosts[dst])
            else:
                self.install_flow(4, priority, match_kwargs, 6)
                self.install_flow(3, priority, match_kwargs, 1)
                self.install_flow(1, priority, match_kwargs, self.top_hosts[dst])

    def get_first_hop_out_port(self, dpid, src, dst, path):
        src_top = self.is_top_host(src)
        dst_top = self.is_top_host(dst)

        if src_top and not dst_top:
            if dpid == 1:
                return 5 if path == "upper" else 6

        elif not src_top and dst_top:
            if dpid == 4:
                return 5 if path == "upper" else 6

        return None

    def send_packet_out(self, msg, out_port):
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        actions = [parser.OFPActionOutput(out_port)]

        data = None
        if msg.buffer_id == ofp.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=msg.match["in_port"],
            actions=actions,
            data=data,
        )

        dp.send_msg(out)

    def send_arp_reply(self, msg, eth_pkt, arp_pkt):
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        target_ip = arp_pkt.dst_ip

        if target_ip not in self.ip_to_mac:
            return

        reply = packet.Packet()

        reply.add_protocol(
            ethernet.ethernet(
                dst=eth_pkt.src,
                src=self.ip_to_mac[target_ip],
                ethertype=0x0806,
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

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath

        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        arp_pkt = pkt.get_protocol(arp.arp)

        if arp_pkt is not None:
            if arp_pkt.opcode == arp.ARP_REQUEST:
                self.send_arp_reply(msg, eth_pkt, arp_pkt)
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)

        if ip_pkt is None or tcp_pkt is None:
            return

        src = ip_pkt.src
        dst = ip_pkt.dst
        tcp_dst = tcp_pkt.dst_port

        # Only dynamically route cross-side TCP traffic.
        if not (
            (self.is_top_host(src) and self.is_bottom_host(dst))
            or
            (self.is_bottom_host(src) and self.is_top_host(dst))
        ):
            return

        flow_size_kb = self.port_to_size_kb.get(tcp_dst, DEFAULT_FLOW_SIZE_KB)

        path = self.choose_path(
            src=src,
            dst=dst,
            tcp_dst=tcp_dst,
            flow_size_kb=flow_size_kb,
        )

        self.install_tcp_path(src, dst, path, tcp_dst)

        self.log_decision(
            src=src,
            dst=dst,
            path=path,
            proto="tcp",
            tcp_dst=tcp_dst,
            flow_size_kb=flow_size_kb,
        )

        out_port = self.get_first_hop_out_port(dp.id, src, dst, path)

        if out_port is not None:
            self.send_packet_out(msg, out_port)

        self.logger.info(
            "[%s] TCP flow src=%s dst=%s tcp_dst=%s size=%.1fKB path=%s",
            self.POLICY_NAME,
            src,
            dst,
            tcp_dst,
            flow_size_kb,
            path,
        )
