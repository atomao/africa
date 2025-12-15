from lightning.pytorch.callbacks import Callback
import torch
from pathlib import Path

class SaveStateDictCallback(Callback):
    """
    Saves LightningModule.model.model.state_dict() 
    after every epoch into a given directory.
    """

    def __init__(self, save_dir: str = "checkpoints_smp", prefix: str = "epoch"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        # Extract SMP model
        if hasattr(pl_module.model, "head"):
            smp_model = pl_module.model.head
        else:
            try:
                smp_model = pl_module.model   # your SMP model
            except AttributeError:
                raise AttributeError(
                    "Expected LightningModule to have pl_module.model.model "
                    "but it does not exist."
                )
        
        epoch = trainer.current_epoch
        save_path = self.save_dir / f"{self.prefix}_{epoch:03d}.pth"

        torch.save(smp_model.state_dict(), save_path)

        # Optional logging
        trainer.logger.log_metrics(
            {"saved_smp_weights_epoch": epoch}, step=trainer.global_step
        )
        print(f"[SaveSMPStateDictCallback] Saved: {save_path}")
