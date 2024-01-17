import time
import pytorch_lightning as pl

class TimeLimitCallback(pl.Callback):
    def __init__(self, max_duration_seconds):
        super().__init__()
        self.max_duration_seconds = max_duration_seconds
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_batch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.max_duration_seconds:
            print(f"Training stopped due to time constraint ({self.max_duration_seconds} seconds).")
            trainer.should_stop = True  # Stops training
