import torch
import time
import json
import os
import pandas as pd
from collections import namedtuple

# 备份原始方法
original_to = torch.Tensor.to

class TransferTracker:
    def __init__(self, filename="transfer_log.jsonl"):
        self.filename = filename
        self.start_time = time.time()
        self.counter = 0
        # 如果文件已存在则删除，防止旧数据干扰
        if os.path.exists(self.filename):
            os.remove(self.filename)
            # create empty file
        open(self.filename, 'w').close()

    def log_advanced(self, name, direction, old_mb, new_mb, dtype_info):
        event = {
            "time": round(time.time() - self.start_time, 4),
            "name": name,
            "dir": direction,
            "old_mb": round(old_mb, 4),
            "new_mb": round(new_mb, 4),
            "diff_mb": round(new_mb - old_mb, 4), # 内存变化量
            "dtype": dtype_info
        }
        with open(self.filename, 'a') as f:
            f.write(json.dumps(event) + "\n")

tracker = TransferTracker()

def patched_to(self, *args, **kwargs):
    source_device = self.device
    source_dtype = self.dtype
    
    # 1. 目标解析
    target_device = source_device
    target_dtype = source_dtype
    
    if args:
        if isinstance(args[0], (torch.device, str)):
            target_device = torch.device(args[0])
        elif isinstance(args[0], torch.dtype):
            target_dtype = args[0]
        elif isinstance(args[0], torch.Tensor):
            target_device = args[0].device
            target_dtype = args[0].dtype
            
    target_device = kwargs.get('device', target_device)
    target_dtype = kwargs.get('dtype', target_dtype)

    # 2. 判定变化类型
    is_dev_change = source_device.type != target_device.type
    is_dtype_change = source_dtype != target_dtype

    if is_dev_change or is_dtype_change:
        name = getattr(self, '_logic_name', f"T_{id(self) % 10000}")
        old_size = self.element_size() * self.nelement()
        
        # 计算新精度下的理论大小
        # 如果只改设备不改精度，new_size == old_size
        dummy_tensor = torch.empty(0, dtype=target_dtype)
        new_size = dummy_tensor.element_size() * self.nelement()
        
        # 记录标签：如果是同设备改精度，标记为 "reformat"
        if not is_dev_change:
            direction = f"{source_device.type}_reformat"
        else:
            direction = f"{source_device.type}->{target_device.type}"
            
        tracker.log_advanced(
            name=name, 
            direction=direction, 
            old_mb=old_size / (1024**2), 
            new_mb=new_size / (1024**2),
            dtype_info=f"{source_dtype}->{target_dtype}"
        )

    return original_to(self, *args, **kwargs)