import time


def run_tests(net):
    """
    Automatically runs traffic experiments inside Mininet.
    """

    print("\n====================================")
    print(" Running Automated Traffic Tests")
    print("====================================\n")

    h1 = net.get("h1")
    h2 = net.get("h2")
    h3 = net.get("h3")
    h4 = net.get("h4")

    print("[1] Running pingall...\n")
    net.pingAll()

    print("\n[2] Sequential traffic tests...\n")

    print(h1.cmd("ping -c 20 10.0.0.5"))
    print(h1.cmd("ping -c 20 10.0.0.6"))

    print("\n[3] Parallel traffic burst...\n")

    h1.sendCmd("ping -c 50 10.0.0.5")
    h2.sendCmd("ping -c 50 10.0.0.6")
    h3.sendCmd("ping -c 50 10.0.0.7")
    h4.sendCmd("ping -c 50 10.0.0.8")

    print("[INFO] Waiting for parallel traffic to finish...\n")

    print(h1.waitOutput())
    print(h2.waitOutput())
    print(h3.waitOutput())
    print(h4.waitOutput())

    print("\n[4] Dumping OpenFlow rules from s1...\n")

    s1 = net.get("s1")
    flows = s1.cmd("ovs-ofctl -O OpenFlow13 dump-flows s1")

    print(flows)

    print("\n====================================")
    print(" Automated Tests Complete")
    print("====================================\n")
