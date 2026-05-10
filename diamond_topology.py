import argparse
import time

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel

from automated_traffic_tests import run_automated_tests
from config import LINK_BW_MBPS, LINK_DELAY


class DiamondTopo(Topo):
    def build(self):
        s1 = self.addSwitch("s1", dpid="0000000000000001")
        s2 = self.addSwitch("s2", dpid="0000000000000002")
        s3 = self.addSwitch("s3", dpid="0000000000000003")
        s4 = self.addSwitch("s4", dpid="0000000000000004")

        for i in range(1, 5):
            h = self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24",
                mac=f"00:00:00:00:00:0{i}",
            )
            self.addLink(h, s1, port2=i)

        for i in range(5, 9):
            h = self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24",
                mac=f"00:00:00:00:00:0{i}",
            )
            self.addLink(h, s4, port2=i - 4)

        # Symmetric diamond multipath links.
        # Upper path: s1 -> s2 -> s4
        # Lower path: s1 -> s3 -> s4
        self.addLink(
            s1,
            s2,
            port1=5,
            port2=1,
            bw=LINK_BW_MBPS,
            delay=LINK_DELAY,
        )

        self.addLink(
            s1,
            s3,
            port1=6,
            port2=1,
            bw=LINK_BW_MBPS,
            delay=LINK_DELAY,
        )

        self.addLink(
            s2,
            s4,
            port1=2,
            port2=5,
            bw=LINK_BW_MBPS,
            delay=LINK_DELAY,
        )

        self.addLink(
            s3,
            s4,
            port1=2,
            port2=6,
            bw=LINK_BW_MBPS,
            delay=LINK_DELAY,
        )


def run(policy_name=None):
    topo = DiamondTopo()

    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=False,
        autoStaticArp=True,
    )

    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)

    print("*** Starting diamond topology")
    net.start()

    for sw in net.switches:
        sw.cmd("ovs-vsctl set Bridge %s protocols=OpenFlow13" % sw.name)

    print("*** Network started")

    time.sleep(3)

    if policy_name:
        run_automated_tests(net, policy_name)

        print()
        print("====================================")
        print(f"{policy_name} experiment complete.")
        print("====================================")
        print()
        print("Mininet will now stop automatically.")
        print()
        print("To rerun this experiment, use:")
        print(f"sudo python3 diamond_topology.py --policy {policy_name}")
        print()

        net.stop()
        return

    print("*** No policy provided.")
    print("*** Dropping into Mininet CLI.")
    print("*** Try: pingall")
    CLI(net)

    print("*** Stopping network")
    net.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["FIFO", "RL"], default=None)
    args = parser.parse_args()

    setLogLevel("info")
    run(policy_name=args.policy)
