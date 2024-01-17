from AGen_model import AGen_model
from videos_dataset import create_video_dataloader

import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import glob
import torch

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        save_last=True)

    wandb.require("service")
    logger = WandbLogger(project=opt.project_name, name=f"{opt.project_name}/test")

    # Set the CUDA_VISIBLE_DEVICES environment variable
    gpu_ids_str = ','.join(map(str, opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    # The trainer is set up using a ddp strategy, meaning each instance of the model processes one batch (one video) in parallel 
    AGen_trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=8000,
        check_val_every_n_epoch=5,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    
    model = AGen_model(opt)
    checkpoint_path = "/outputs/checkpoints/last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))

    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)

    testset = create_video_dataloader(opt.videos_dataset.test)
    
    AGen_trainer.test(model, testset, ckpt_path=checkpoint_path)


if __name__ == '__main__':
    main()