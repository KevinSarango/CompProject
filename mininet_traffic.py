import time

from config import PINGALL_ATTEMPTS, PINGALL_RETRY_WAIT_SECONDS


def run_pingall_with_retries(net):
    print()
    print("====================================")
    print(" Checking connectivity with pingall")
    print("====================================")

    for attempt in range(1, PINGALL_ATTEMPTS + 1):
        print(f"[PINGALL] Attempt {attempt}/{PINGALL_ATTEMPTS}")

        loss = net.pingAll()

        try:
            loss_value = float(loss)
        except (TypeError, ValueError):
            loss_value = 100.0

        if loss_value == 0.0:
            print("[PINGALL] Success: 0% packet loss")
            return True

        print(f"[PINGALL] Failed attempt {attempt}: {loss_value}% packet loss")

        if attempt < PINGALL_ATTEMPTS:
            print(f"[PINGALL] Retrying in {PINGALL_RETRY_WAIT_SECONDS} seconds...")
            time.sleep(PINGALL_RETRY_WAIT_SECONDS)

    print("[PINGALL] Failed after all attempts.")
    return False


def run_tests(net):
    """
    Automatically runs simple traffic experiments inside Mininet.

    This file is no longer the main TrafPy experiment path, but it is kept
    for manual/basic testing.
    """

    print("\n====================================")
    print(" Running Automated Traffic Tests")
    print("====================================\n")

    h1 = net.get("h1")
    h2 = net.get("h2")
    h3 = net.get("h3")
    h4 = net.get("h4")

    if not run_pingall_with_retries(net):
        print("[ERROR] pingall failed. Exiting basic traffic test.")
        return False

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

    return True
