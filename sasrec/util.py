import time
import torch
import threading
import subprocess


def check_gpu_utilization():
    if not torch.cuda.is_available():
        return
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        utils = [
            int(x.strip())
            for x in result.stdout.strip().split("\n")
            if x.strip().isdigit()
        ]
        if utils:
            max_util = max(utils)
            if max_util > 50:
                print(
                    "\n=================================================================\n"
                    f"WARNING: GPU utilization is already over 50% ({max_util}%) before tests!"
                    "\n=================================================================\n"
                )
    except Exception as e:
        print(f"Failed to check GPU utilization: {e}")


class GpuMemoryTracker:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.keep_running = False
        self.thread = None
        self.baseline_used = 0
        self.min_free = 0
        self.total_mem = 0
        self.peak_delta_mb = 0.0

    def _monitor(self):
        while self.keep_running:
            free_mem, _ = torch.cuda.mem_get_info()
            if free_mem < self.min_free:
                self.min_free = free_mem
            time.sleep(self.interval)

    def __enter__(self):
        if not torch.cuda.is_available():
            return self

        free_mem, total_mem = torch.cuda.mem_get_info()
        self.total_mem = total_mem
        self.baseline_used = total_mem - free_mem
        self.min_free = free_mem

        self.keep_running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not torch.cuda.is_available():
            return

        self.keep_running = False
        if self.thread:
            self.thread.join()

        peak_used = self.total_mem - self.min_free
        peak_allocated_delta = peak_used - self.baseline_used
        self.peak_delta_mb = peak_allocated_delta / (1024**2)
