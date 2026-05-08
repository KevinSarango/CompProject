from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, arp
from rl_policy import choose_path


class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.installed = False

        self.ip_to_mac = {
            f"10.0.0.{i}": f"00:00:00:00:00:0{i}" for i in range(1, 9)
        }

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp

        parser = dp.ofproto_parser
        ofp = dp.ofproto

        # Table-miss sends ARP to controller.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.add_flow(dp, 0, match, actions)

        self.logger.info("Switch connected: s%s", dp.id)

        if len(self.datapaths) == 4 and not self.installed:
            self.install_all_paths()
            self.installed = True

    def add_flow(self, dp, priority, match, actions):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
        )
        dp.send_msg(mod)

    def install_ipv4(self, dpid, src_ip, dst_ip, out_port):
        dp = self.datapaths[dpid]
        parser = dp.ofproto_parser

        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=src_ip,
            ipv4_dst=dst_ip,
        )
        actions = [parser.OFPActionOutput(out_port)]
        self.add_flow(dp, 100, match, actions)

    def choose_path(self, src_ip, dst_ip):
        """
        Basic multipath policy.
        Even destination IPs use s1-s2-s4.
        Odd destination IPs use s1-s3-s4.
        Replace this later with a loaded offline RL policy.
        """
        dst_num = int(dst_ip.split(".")[-1])
        return "upper" if dst_num % 2 == 0 else "lower"

    def install_all_paths(self):
        self.logger.info("Installing deterministic multipath IPv4 rules")

        top_hosts = {
            "10.0.0.1": 1,
            "10.0.0.2": 2,
            "10.0.0.3": 3,
            "10.0.0.4": 4,
        }

        bottom_hosts = {
            "10.0.0.5": 1,
            "10.0.0.6": 2,
            "10.0.0.7": 3,
            "10.0.0.8": 4,
        }

        all_hosts = {**top_hosts, **bottom_hosts}

        for src_ip in all_hosts:
            for dst_ip in all_hosts:
                if src_ip == dst_ip:
                    continue

                src_num = int(src_ip.split(".")[-1])
                dst_num = int(dst_ip.split(".")[-1])

                src_top = src_num <= 4
                dst_top = dst_num <= 4

                # Same side on s1
                if src_top and dst_top:
                    self.install_ipv4(1, src_ip, dst_ip, top_hosts[dst_ip])

                # Same side on s4
                elif not src_top and not dst_top:
                    self.install_ipv4(4, src_ip, dst_ip, bottom_hosts[dst_ip])

                # Top to bottom
                elif src_top and not dst_top:
                    path = choose_path(src, dst)

                    if path == "upper":
                        # s1 -> s2 -> s4
                        self.install_ipv4(1, src_ip, dst_ip, 5)
                        self.install_ipv4(2, src_ip, dst_ip, 2)
                        self.install_ipv4(4, src_ip, dst_ip, bottom_hosts[dst_ip])
                    else:
                        # s1 -> s3 -> s4
                        self.install_ipv4(1, src_ip, dst_ip, 6)
                        self.install_ipv4(3, src_ip, dst_ip, 2)
                        self.install_ipv4(4, src_ip, dst_ip, bottom_hosts[dst_ip])

                # Bottom to top
                else:
                    path = choose_path(src, dst)

                    if path == "upper":
                        # s4 -> s2 -> s1
                        self.install_ipv4(4, src_ip, dst_ip, 5)
                        self.install_ipv4(2, src_ip, dst_ip, 1)
                        self.install_ipv4(1, src_ip, dst_ip, top_hosts[dst_ip])
                    else:
                        # s4 -> s3 -> s1
                        self.install_ipv4(4, src_ip, dst_ip, 6)
                        self.install_ipv4(3, src_ip, dst_ip, 1)
                        self.install_ipv4(1, src_ip, dst_ip, top_hosts[dst_ip])

        self.logger.info("All IPv4 multipath rules installed")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Only answer ARP.
        IPv4 forwarding is handled by static flows.
        """
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        arp_pkt = pkt.get_protocol(arp.arp)

        if arp_pkt is None:
            return

        if arp_pkt.opcode != arp.ARP_REQUEST:
            return

        target_ip = arp_pkt.dst_ip

        if target_ip not in self.ip_to_mac:
            return

        src_mac = self.ip_to_mac[target_ip]

        e = ethernet.ethernet(
            dst=eth.src,
            src=src_mac,
            ethertype=0x0806,
        )

        a = arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=src_mac,
            src_ip=target_ip,
            dst_mac=arp_pkt.src_mac,
            dst_ip=arp_pkt.src_ip,
        )

        reply = packet.Packet()
        reply.add_protocol(e)
        reply.add_protocol(a)
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