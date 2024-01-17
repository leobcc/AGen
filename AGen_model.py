from VideoReconstructionModel.code.v2a_model import V2AModel
from lib.model.v2a import V2A
from VideoReconstructionModel.code.lib.model.networks import ImplicitNet, RenderingNet, GeometryEncodingNet
from VideoReconstructionModel.code.lib.datasets import create_dataset
from Callbacks import TimeLimitCallback 

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import models
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import glob
import yaml

from VideoReconstructionModel.code.lib.datasets.pretrained_encoders import EncodingBackbone

class AGen_model(pl.LightningModule):
    def __init__(self, opt):
        super(AGen_model, self).__init__()

        # Configuration options
        self.opt = opt
        self.incremented_on_epoch = 1   # Keeps track of the epoch on which the sampling has been incremented last

        # Geometry encoding network
        if self.opt.model.AGen.use_geometry_encoder == True:
            self.geometry_encoding_network = GeometryEncodingNet()
        else:
            self.geometry_encoding_network = None

        # Implicit network
        self.implicit_network = ImplicitNet(opt.model.implicit_network)

        # Rendering network
        self.rendering_network = RenderingNet(opt.model.rendering_network)

        # Rendering encoding network
        #self.rendering_encoding_network = EncodingNet(input_dim=2048, output_dim=256)

    def training_step(self, batch):
        # Setting the configuration file for the video in the batch ------------------------------
        # Each batch contains the path to one training video
        # TODO: This is probably useless, but it shouldn't hurt 
        torch.cuda.set_device(self.device)
        video_path = batch[0]   # Each batch consists of one video
        
        metainfo_path = os.path.join(video_path, 'confs/', 'video_metainfo.yaml')
        with open(metainfo_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
            self.opt.dataset.metainfo = loaded_config.get('metainfo', {})

        # TODO: check all the things that need to be turned off or on if the training is not on the first epoch
        video_outputs_folder = os.path.abspath(os.path.join(os.getcwd(), 'Video/', self.opt.dataset.metainfo.data_dir))
        video_checkpoints_folder = os.path.abspath(os.path.join(os.getcwd(), 'Video/', self.opt.dataset.metainfo.data_dir, 'checkpoints'))   # Outputs folder of the video 
        #if os.path.exists(video_checkpoints_folder):
        #    self.opt.model.smpl_init = False   # Turn off the initialization with the smpl weights 
        
        # Incremental sampling -------------------------------------------------------------------
        if self.opt.model.incremental_sampling == True:
            if self.current_epoch >=1 and self.incremented_on_epoch != self.current_epoch and self.current_epoch <= (1+self.opt.model.incremental_sampling_steps):   # If the sampling has not been incremented yet on this epoch
                if self.opt.model.increment_profile == "Squared":
                    self.opt.dataset.train.num_sample = int(self.opt.dataset.train.num_sample*2)
                    self.opt.model.ray_sampler.N_samples = int(self.opt.model.ray_sampler.N_samples/2)
                    self.opt.model.ray_sampler.N_samples_eval = int(self.opt.model.ray_sampler.N_samples_eval/2)
                    self.opt.model.ray_sampler.N_samples_extra = int(self.opt.model.ray_sampler.N_samples_extra/2)
                if self.opt.model.increment_profile == "Linear":
                    if self.opt.model.incremental_sampling_steps > 3:
                        raise ValueError("The training will result in a negative number of samples, please adjust the values in train.py accordingly.")
                    self.opt.dataset.train.num_sample = int(self.opt.dataset.train.num_sample + 1024)
                    self.opt.model.ray_sampler.N_samples = int(self.opt.model.ray_sampler.N_samples - 16)
                    self.opt.model.ray_sampler.N_samples_eval = int(self.opt.model.ray_sampler.N_samples_eval - 32)
                    self.opt.model.ray_sampler.N_samples_extra = int(self.opt.model.ray_sampler.N_samples_extra - 8)
        
        # Video reconstruction step --------------------------------------------------------------
        '''During this step, video reconstruction is performed on each single video, resulting in 
        the training of the implicit network, the rendering network and the relative encodings
        '''  
        # Callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=video_checkpoints_folder,
            filename="{epoch:04d}-{loss}",
            save_on_train_epoch_end=True,
            save_last=True)
        time_limit_callback = TimeLimitCallback(max_duration_seconds=self.opt.max_duration_seconds)

        # Initialize the Dataloaders
        trainset = create_dataset(self.opt.dataset.metainfo, self.opt.dataset.train)   # Dataloader for the frames of each video in the trainset
        validset = create_dataset(self.opt.dataset.metainfo, self.opt.dataset.valid)   # Dataloader for the frames of each video in the trainset

        # Initialize the VideoReconstructionModel
        if os.path.exists(video_checkpoints_folder):   # If a last.ckpt checkpoint is available it will be used to initialize part of the model
            checkpoint = os.path.join(video_checkpoints_folder, 'last.ckpt')
            v2a_trainer = pl.Trainer(
                gpus=[self.device.index],   # The trainer is set up such that the batch (video) is processed on the assigned gpu
                accelerator="gpu",
                callbacks=[checkpoint_callback, time_limit_callback],
                max_epochs=8000,
                logger=self.logger,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                resume_from_checkpoint=checkpoint,
                enable_progress_bar=True,
                enable_model_summary=False
            ) 
            model = V2AModel.load_from_checkpoint(checkpoint, opt=self.opt, implicit_network=self.implicit_network, 
                                                                            geometry_encoding_network=self.geometry_encoding_network,
                                                                            rendering_network=self.rendering_network)   # Load the model from the previous checkpoint
            model.model.implicit_network = self.implicit_network   # Update the implicit network to the current state
            if model.model.geometry_encoding_network is not None:
                model.model.geometry_encoding_network = self.geometry_encoding_network   # Update the implicit network to the current state
            model.model.rendering_network = self.rendering_network   # Update the rendering netwokr to the current state
        else:   # If the video has never been processed before it will be initialized from scratch
            v2a_trainer = pl.Trainer(
                gpus=[self.device.index],   # The trainer is set up such that the batch (video) is processed on the assigned gpu
                accelerator="gpu",
                callbacks=[checkpoint_callback, time_limit_callback],
                max_epochs=8000,
                logger=self.logger,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                enable_model_summary=False
            ) 
            model = V2AModel(self.opt, implicit_network=self.implicit_network, 
                                        geometry_encoding_network = self.geometry_encoding_network,  
                                        rendering_network=self.rendering_network)   # Instance of the VideoReconstruction model        
    
        # Train the model over the video in the batch
        v2a_trainer.fit(model, trainset)

        # Inference on the V2AModel after fitting ------------------------------------------------

        # Validation metrics over videos in the trainset -----------------------------------------
        v2a_trainer.validate(model, validset)   # Run evaluation on the trainset (the validset here is the frame dataloader for validation)
        validation_metrics = v2a_trainer.callback_metrics

        return 
    
    def configure_optimizers(self):
        # Define your optimizer(s) here
        
        # Example optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # Validation -----------------------------------------------------------------------------
    '''This validation is performed at inference time on the unseen videos in the validset'''
    def validation_step(self, batch, *args, **kwargs):
        # Setting the configuration file for the video in the batch ------------------------------
        # Each batch contains the path to one training video
        # TODO: This is probably useless, but it shouldn't hurt 
        torch.cuda.set_device(self.device)
        video_path = batch[0]   # Each batch consists of one video
        
        metainfo_path = os.path.join(video_path, 'confs/', 'video_metainfo.yaml')
        with open(metainfo_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
            self.opt.dataset.metainfo = loaded_config.get('metainfo', {})

        video_outputs_folder = os.path.abspath(os.path.join(os.getcwd(), 'Video/', self.opt.dataset.metainfo.data_dir))
        video_checkpoints_folder = os.path.abspath(os.path.join(os.getcwd(), 'Video/', self.opt.dataset.metainfo.data_dir, 'checkpoints'))   # Outputs folder of the video 
        #if os.path.exists(video_checkpoints_folder):
        #    self.opt.model.smpl_init = False   # Turn off the initialization with the smpl weights 

        # Validation on never-seen validset: inference time --------------------------------------
        # Initialize the Dataloaders
        trainset = create_dataset(self.opt.dataset.metainfo, self.opt.dataset.train)   # Dataloader for the frames of each video in the validset
        validset = create_dataset(self.opt.dataset.metainfo, self.opt.dataset.valid)   # Dataloader for the frames of each video in the validset
        
        # Initialize the VideoReconstructionModel ------------------------------------------------
        # Callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=video_checkpoints_folder,
            filename="{epoch:04d}-{loss}",
            save_on_train_epoch_end=True,
            save_last=True)
        time_limit_callback = TimeLimitCallback(max_duration_seconds=self.opt.max_duration_seconds)

        v2a_trainer = pl.Trainer(
                gpus=[self.device.index],   # The trainer is set up such that the batch (video) is processed on the assigned gpu
                accelerator="gpu",
                callbacks=[checkpoint_callback, time_limit_callback],
                max_epochs=8000,
                logger=self.logger,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                enable_model_summary=False
            ) 
        model = V2AModel(self.opt, implicit_network=self.implicit_network, 
                                    geometry_encoding_network = self.geometry_encoding_network,
                                    rendering_network=self.rendering_network)   # Instance of the VideoReconstruction model

        # Setting all the networks to evaluation mode --------------------------------------------
        self.implicit_network.eval()
        if self.geometry_encoding_network is not None:
            self.geometry_encoding_network.eval()

        self.rendering_network.eval()
        #self.rendering_encoding_network.eval()

        model.eval()
        
        # Validation metrics over videos in the validset -----------------------------------------
        v2a_trainer.validate(model, validset)   # Run evaluation on the validset (the validset here is the frame dataloader for validation)
        validation_metrics = v2a_trainer.callback_metrics

        # Validation on never-seen validset: reconstruction time ---------------------------------
        # TODO: reconstruction option, measure the time needed, or constraint it to a maximum time

        return 

    def validation_step_end(self, outputs):
        # Validation step end
        pass

    def validation_epoch_end(self, outputs):
        # Validation epoch end
        pass

    # Test -----------------------------------------------------------------------------------
    def test_step(self, batch, *args, **kwargs):
        torch.cuda.set_device(self.device)
        video_path = batch[0]   # Each batch consists of one video
        
        metainfo_path = os.path.join(video_path, 'confs/', 'video_metainfo.yaml')
        with open(metainfo_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
            self.opt.dataset.metainfo = loaded_config.get('metainfo', {})

        video_outputs_folder = os.path.abspath(os.path.join(os.getcwd(), 'Video/', self.opt.dataset.metainfo.data_dir))
        video_checkpoints_folder = os.path.abspath(os.path.join(os.getcwd(), 'Video/', self.opt.dataset.metainfo.data_dir, 'checkpoints'))   # Outputs folder of the video 
        
        # Validation on never-seen validset: inference time --------------------------------------
        # Initialize the Dataloaders
        trainset = create_dataset(self.opt.dataset.metainfo, self.opt.dataset.train)   # Dataloader for the frames of each video in the testset
        validset = create_dataset(self.opt.dataset.metainfo, self.opt.dataset.valid)   # Dataloader for the frames of each video in the testset
        testset = create_dataset(self.opt.dataset.metainfo, self.opt.dataset.test)   # Dataloader for the frames of each video in the testset
        
        # Initialize the VideoReconstructionModel ------------------------------------------------
        # Callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=video_checkpoints_folder,
            filename="{epoch:04d}-{loss}",
            save_on_train_epoch_end=True,
            save_last=True)
        #time_limit_callback = TimeLimitCallback(max_duration_seconds=self.opt.max_duration_seconds)

        v2a_trainer = pl.Trainer(
                gpus=[self.device.index],   # The trainer is set up such that the batch (video) is processed on the assigned gpu
                accelerator="gpu",
                callbacks=[checkpoint_callback],
                max_epochs=self.opt.refinement_epochs,
                logger=self.logger,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                enable_model_summary=False
            ) 
        # Here the model is initialized with the pre-trained networks, that have been loaded from the AGen model
        model = V2AModel(self.opt, implicit_network=self.implicit_network, 
                                    geometry_encoding_network = self.geometry_encoding_network,
                                    rendering_network=self.rendering_network)

        if self.opt.dataset.test.pretrained == False:
            # Re-initialize the networks ---------------------------------------------------------
            self.implicit_network = ImplicitNet(self.opt.model.implicit_network)
            if self.geometry_encoding_network is not None:
                self.geometry_encoding_network = GeometryEncodingNet()
            else:
                self.geometry_encoding_network = None
            self.rendering_network = RenderingNet(self.opt.model.rendering_network)
            #self.rendering_encoding_network = EncodingNet(input_dim=2048, output_dim=256)

            model = V2AModel(self.opt, implicit_network=self.implicit_network,
                                        geometry_encoding_network = self.geometry_encoding_network, 
                                        rendering_network=self.rendering_network)

            # Flag the type of test as the one of the non-pretrained model -----------------------
            self.opt.dataset.metainfo.type = "test-non-pretrained"
            '''This is not gonna work with the additional encoding networks, since it doesn't reaally represent the non-pretrained baseline model'''
        else:
            sample_from_mesh = False
            if sample_from_mesh == True:
                model.eval()
                sampled_points = model.sampling_from_mesh(trainset, self.opt.dataset.train)
                print("sampled points len:", len(sampled_points))
                print("sampled points shape:", sampled_points[0].shape)
                trainset = create_dataset(self.opt.dataset.metainfo, self.opt.dataset.train, pretrained_uv=sampled_points)   

        if self.opt.dataset.test.mode == "short_time":   # Refining the reconstruction with a short-time optimization
            # Train the model over the video in the batch
            model.train()
            v2a_trainer.fit(model, trainset)   # Fit the model on the trainset (the trainset here is the frame dataloader for the testset)
         
        # Setting all the networks to evaluation mode --------------------------------------------
        self.implicit_network.eval()
        if self.geometry_encoding_network is not None:
            self.geometry_encoding_network.eval()

        self.rendering_network.eval()
        #self.rendering_encoding_network.eval()

        model.eval()

        if self.opt.dataset.test.size == "tiny":
            # Validation metrics over videos in the testset ------------------------------------------
            v2a_trainer.validate(model, validset)   # Run evaluation on the testset (the validset here is the frame dataloader for the testset)
            validation_metrics = v2a_trainer.callback_metrics
        elif self.opt.dataset.test.size == "reduced":   # Here the dataloader has already been reduced to the frames in the testset when it was created
            v2a_trainer.test(model, testset)   # Run evaluation on the testset (42 frames of 0-941)
        elif self.opt.dataset.test.size == "full":
            v2a_trainer.test(model, testset)   # Run evaluation on the testset (the validset here is the frame dataloader for the testset)

        return