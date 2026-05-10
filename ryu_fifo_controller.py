from ryu_common import BaseMultipathController


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "FIFO"
    METRICS_FILE = "data/fifo_metrics.csv"

    def choose_path(self, src, dst, tcp_dst=None, flow_size_kb=None):
        """
        FIFO / round-robin baseline.

        Every new TCP flow alternates between the upper and lower path.
        """
        path = "upper" if self.flow_counter % 2 == 0 else "lower"
        self.flow_counter += 1
        return path
