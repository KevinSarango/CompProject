from config import FIFO_METRICS_FILE, NUM_PATHS, PATHS
from ryu_common import BaseMultipathController


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "FIFO"
    METRICS_FILE = FIFO_METRICS_FILE

    def choose_path(self, src, dst, flow_info=None):
        path = PATHS[self.flow_counter % NUM_PATHS]
        self.flow_counter += 1
        return path
