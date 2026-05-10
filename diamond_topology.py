import argparse
import sys
import time

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel

from automated_traffic_tests import run_automated_tests


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

        # Diamond multipath links.
        self.addLink(s1, s2, port1=5, port2=1, bw=5, delay="5ms")
        self.addLink(s1, s3, port1=6, port2=1, bw=5, delay="5ms")
        self.addLink(s2, s4, port1=2, port2=5, bw=5, delay="5ms")
        self.addLink(s3, s4, port1=2, port2=6, bw=5, delay="5ms")


def verify_connectivity(net, max_attempts=3):
    """
    Run pingall up to max_attempts times.

    If all hosts connect, continue.
    If pingall fails after max_attempts, stop the network and exit.
    """

    print()
    print("====================================")
    print(" Verifying host connectivity")
    print("====================================")

    for attempt in range(1, max_attempts + 1):
        print(f"*** pingall attempt {attempt}/{max_attempts}")

        packet_loss_percent = net.pingAll()

        if packet_loss_percent == 0:
            print("*** Connectivity verified: 0% packet loss")
            print("====================================")
            print()
            return True

        print(f"*** Connectivity failed: {packet_loss_percent}% packet loss")

        if attempt < max_attempts:
            print("*** Waiting before retry...")
            time.sleep(2)

    print()
    print("[ERROR] Host connectivity failed after 3 pingall attempts.")
    print("[ERROR] Stopping Mininet and exiting.")
    print("====================================")
    print()

    net.stop()
    sys.exit(1)


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

    # Give Ryu time to connect switches and install rules.
    time.sleep(3)

    # Verify all hosts can reach each other before running traffic tests.
    verify_connectivity(net, max_attempts=3)

    if policy_name:
        run_automated_tests(net, policy_name)

    print("*** Dropping into Mininet CLI")
    CLI(net)

    print("*** Stopping network")
    net.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["FIFO", "RL"], default=None)
    args = parser.parse_args()

    setLogLevel("info")
    run(policy_name=args.policy)
