import argparse
import os
import time

# Force this topology process to use three-path output filenames.
os.environ["TOPO_MODE"] = "three_path"

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel

from automated_traffic_tests import run_automated_tests
from config import THREE_PATH_LINKS


class ThreePathTopo(Topo):
    """
    Three-path multipath topology.

              h1 h2 h3 h4
                    |
                   s1
              /     |     \
            s2      s3      s5
              \     |     /
                   s4
                    |
              h5 h6 h7 h8

    Paths:
        low_delay: s1 -> s2 -> s4
        balanced:  s1 -> s3 -> s4
        high_bw:   s1 -> s5 -> s4
    """

    def build(self):
        s1 = self.addSwitch("s1", dpid="0000000000000001")
        s2 = self.addSwitch("s2", dpid="0000000000000002")
        s3 = self.addSwitch("s3", dpid="0000000000000003")
        s4 = self.addSwitch("s4", dpid="0000000000000004")
        s5 = self.addSwitch("s5", dpid="0000000000000005")

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

        low = THREE_PATH_LINKS["low_delay"]
        balanced = THREE_PATH_LINKS["balanced"]
        high = THREE_PATH_LINKS["high_bw"]

        # Path 0: low delay, low bandwidth
        self.addLink(
            s1,
            s2,
            port1=5,
            port2=1,
            bw=low["bw"],
            delay=low["delay"],
        )
        self.addLink(
            s2,
            s4,
            port1=2,
            port2=5,
            bw=low["bw"],
            delay=low["delay"],
        )

        # Path 1: balanced
        self.addLink(
            s1,
            s3,
            port1=6,
            port2=1,
            bw=balanced["bw"],
            delay=balanced["delay"],
        )
        self.addLink(
            s3,
            s4,
            port1=2,
            port2=6,
            bw=balanced["bw"],
            delay=balanced["delay"],
        )

        # Path 2: high bandwidth, high delay
        self.addLink(
            s1,
            s5,
            port1=7,
            port2=1,
            bw=high["bw"],
            delay=high["delay"],
        )
        self.addLink(
            s5,
            s4,
            port1=2,
            port2=7,
            bw=high["bw"],
            delay=high["delay"],
        )


def run(policy_name=None):
    topo = ThreePathTopo()

    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=False,
        autoStaticArp=True,
    )

    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)

    print("*** Starting three-path topology")
    net.start()

    for sw in net.switches:
        sw.cmd("ovs-vsctl set Bridge %s protocols=OpenFlow13" % sw.name)

    print("*** Network started")
    time.sleep(3)

    if policy_name:
        run_automated_tests(net, policy_name)
        print()
        print("====================================")
        print(f"{policy_name} three-path experiment complete.")
        print("====================================")
        net.stop()
        return

    print("*** No policy provided.")
    print("*** Dropping into Mininet CLI.")
    CLI(net)

    print("*** Stopping network")
    net.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["FIFO", "RL"], default=None)
    args = parser.parse_args()

    setLogLevel("info")
    run(policy_name=args.policy)
