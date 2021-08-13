import os, sys, gc
import psutil
 
from mlagents.torch_utils.torch import torch

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
        print("[DEBUG] system version: ", sys.version)
        print("[DEBUG] CPU percentage: ", psutil.cpu_percent())
        print("[DEBUG] CPU virtual memory: ", psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('[DEBUG] memory GB: ', memoryUse)


if __name__ == "__main__":
    cpuStats()
    memReport()
