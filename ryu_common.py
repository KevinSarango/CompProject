import csv
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, arp


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

        os.makedirs("data", exist_ok=True)
        self.init_metrics_file()

    def init_metrics_file(self):
        with open(self.METRICS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "policy", "src", "dst", "path"])

    def log_decision(self, src, dst, path):
        with open(self.METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), self.POLICY_NAME, src, dst, path])

    def choose_path(self, src, dst):
        raise NotImplementedError("Subclasses must implement choose_path")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp

        ofp = dp.ofproto
        parser = dp.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.add_flow(dp, 0, match, actions)

        self.logger.info("[%s] Switch connected: s%s", self.POLICY_NAME, dp.id)

        if len(self.datapaths) == 4 and not self.installed:
            self.install_all_paths()
            self.installed = True

    def add_flow(self, dp, priority, match, actions):
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
        )
        dp.send_msg(mod)

    def install_ipv4(self, dpid, src, dst, out_port):
        dp = self.datapaths[dpid]
        parser = dp.ofproto_parser

        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=src,
            ipv4_dst=dst,
        )
        actions = [parser.OFPActionOutput(out_port)]
        self.add_flow(dp, 100, match, actions)

    def install_all_paths(self):
        hosts = list(self.top_hosts.keys()) + list(self.bottom_hosts.keys())

        for src in hosts:
            for dst in hosts:
                if src == dst:
                    continue

                src_num = int(src.split(".")[-1])
                dst_num = int(dst.split(".")[-1])

                src_top = src_num <= 4
                dst_top = dst_num <= 4

                if src_top and dst_top:
                    self.install_ipv4(1, src, dst, self.top_hosts[dst])

                elif not src_top and not dst_top:
                    self.install_ipv4(4, src, dst, self.bottom_hosts[dst])

                elif src_top and not dst_top:
                    path = self.choose_path(src, dst)
                    self.log_decision(src, dst, path)

                    if path == "upper":
                        self.install_ipv4(1, src, dst, 5)
                        self.install_ipv4(2, src, dst, 2)
                        self.install_ipv4(4, src, dst, self.bottom_hosts[dst])
                    else:
                        self.install_ipv4(1, src, dst, 6)
                        self.install_ipv4(3, src, dst, 2)
                        self.install_ipv4(4, src, dst, self.bottom_hosts[dst])

                else:
                    reverse_path = self.choose_path(dst, src)
                    self.log_decision(src, dst, reverse_path)

                    if reverse_path == "upper":
                        self.install_ipv4(4, src, dst, 5)
                        self.install_ipv4(2, src, dst, 1)
                        self.install_ipv4(1, src, dst, self.top_hosts[dst])
                    else:
                        self.install_ipv4(4, src, dst, 6)
                        self.install_ipv4(3, src, dst, 1)
                        self.install_ipv4(1, src, dst, self.top_hosts[dst])

        self.logger.info("[%s] Installed all IPv4 multipath rules", self.POLICY_NAME)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        arp_pkt = pkt.get_protocol(arp.arp)

        if arp_pkt is None or arp_pkt.opcode != arp.ARP_REQUEST:
            return

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
