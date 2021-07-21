from torch.utils.tensorboard import SummaryWriter
"""
Global Logging file to enable logging from anywhere
"""

def init(log_dir):
    global writer
    writer = SummaryWriter(log_dir)

