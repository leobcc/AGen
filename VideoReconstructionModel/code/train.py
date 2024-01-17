from v2a_model import V2AModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob

@hydra.main(config_path="/AGen/confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        save_last=True)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    if not opt.model.incremental_sampling:
        trainer = pl.Trainer(
            gpus=1,
            accelerator="gpu",
            callbacks=[checkpoint_callback],
            max_epochs=8000,
            check_val_every_n_epoch=50,
            logger=logger,
            log_every_n_steps=1,
            num_sanity_val_steps=0
        )

        model = V2AModel(opt)
        trainset = create_dataset(opt.dataset.metainfo, opt.dataset.train)
        validset = create_dataset(opt.dataset.metainfo, opt.dataset.valid)

        if opt.model.is_continue == True:
            checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
            trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
        else: 
            trainer.fit(model, trainset, validset)

    else:
        # First iteration of training: initial sampling values
        trainer = pl.Trainer(
            gpus=1,
            accelerator="gpu",
            callbacks=[checkpoint_callback],
            max_epochs=opt.model.epochs_increment_interval,
            check_val_every_n_epoch=50,
            logger=logger,
            log_every_n_steps=1,
            num_sanity_val_steps=0
        )

        model = V2AModel(opt)
        trainset = create_dataset(opt.dataset.metainfo, opt.dataset.train)
        validset = create_dataset(opt.dataset.metainfo, opt.dataset.valid)

        if opt.model.is_continue == True:
            checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
            trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
        else: 
            trainer.fit(model, trainset, validset)

        # Further iterations of training: incremental values
        for i in range(opt.model.incremental_sampling_steps):
            if opt.model.increment_profile == "Squared":
                opt.dataset.train.num_sample = int(opt.dataset.train.num_sample*2)
                opt.model.ray_sampler.N_samples = int(opt.model.ray_sampler.N_samples/2)
                opt.model.ray_sampler.N_samples_eval = int(opt.model.ray_sampler.N_samples_eval/2)
                opt.model.ray_sampler.N_samples_extra = int(opt.model.ray_sampler.N_samples_extra/2)
            if opt.model.increment_profile == "Linear":
                if opt.model.incremental_sampling_steps > 3:
                    raise ValueError("The training will result in a negative number of samples, please adjust the values in train.py accordingly.")
                opt.dataset.train.num_sample = int(opt.dataset.train.num_sample + 1024)
                opt.model.ray_sampler.N_samples = int(opt.model.ray_sampler.N_samples - 16)
                opt.model.ray_sampler.N_samples_eval = int(opt.model.ray_sampler.N_samples_eval - 32)
                opt.model.ray_sampler.N_samples_extra = int(opt.model.ray_sampler.N_samples_extra - 8)

            trainer = pl.Trainer(
                gpus=1,
                accelerator="gpu",
                callbacks=[checkpoint_callback],
                max_epochs=opt.model.epochs_increment_interval+(i+1)*opt.model.epochs_increment_interval,
                check_val_every_n_epoch=50,
                logger=logger,
                log_every_n_steps=1,
                num_sanity_val_steps=0
            )

            model = V2AModel(opt) # Initialize the model with the new confs values
            trainset = create_dataset(opt.dataset.metainfo, opt.dataset.train)
            validset = create_dataset(opt.dataset.metainfo, opt.dataset.valid)

            checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
            trainer.fit(model, trainset, validset, ckpt_path=checkpoint)

        # Continue the training further
        trainer = pl.Trainer(
            gpus=1,
            accelerator="gpu",
            callbacks=[checkpoint_callback],
            max_epochs=8000,
            check_val_every_n_epoch=50,
            logger=logger,
            log_every_n_steps=1,
            num_sanity_val_steps=0
        )

        model = V2AModel(opt) # Initialize the model with the new confs values
        trainset = create_dataset(opt.dataset.metainfo, opt.dataset.train)
        validset = create_dataset(opt.dataset.metainfo, opt.dataset.valid)

        checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)

if __name__ == '__main__':
    main()