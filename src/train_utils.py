"""
functions for initialisation model and trainer
"""

import os
import copy

from omegaconf import OmegaConf, open_dict

import pytorch_lightning as ptl

from nemo.utils import logging, exp_manager

def init_model(
        model,
        train_manifest_dir: str,
        valid_manifest_dir: str,
        train_batch_size: int = 1,
        valid_batch_size: int = 1,
        learning_rate: float = 3e-5,
    ):
    """
    init model config parameters

    Parameters:
    -----------
        model:
            model for initialisation it's config

        train_manifest_dir (str):
            path to train manifest file
        
        valid_manifest_dir (str):
            path to valid manifest file

        train_batch_size (int):
            number of samples in batch to train on

        valid_batch_size (int):
            number of samples in batch for validation

        learning_rate (float):
            starting learning rate
    
    Return:
    -------
        model for training

    """

    cfg = copy.deepcopy(model.cfg)

    audio_augmentations = dict(
    white_noise = dict(
        prob=0.5,
        min_level=-90,
        max_level=-46
    ),
    gain = dict(
        prob=0.5,
        min_gain_dbfs=0,
        max_gain_dbfs=50
    ))

    with open_dict(cfg):
        ## TRAIN CONFIG ##
        cfg.train_ds.manifest_filepath = train_manifest_dir
        cfg.train_ds.normalize_transcripts = False
        cfg.train_ds.batch_size = train_batch_size
        cfg.train_ds.num_workers = 2
        cfg.train_ds.pin_memory = True
        cfg.train_ds.trim_silence = True
        cfg.train_ds.sample_rate = 16000

        ## VALID CONFIG ##
        cfg.validation_ds.manifest_filepath = valid_manifest_dir
        cfg.validation_ds.normalize_transcripts = False
        cfg.validation_ds.batch_size = valid_batch_size
        cfg.validation_ds.num_workers = 2
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.trim_silence = True
        cfg.validation_ds.sample_rate = 16000
        # cfg.train_ds.augmentor = audio_augmentations

    # setup data loaders with new configs
    model.setup_training_data(cfg.train_ds)
    model.setup_multiple_validation_data(cfg.validation_ds)

    ## OPTIMIZERS ##
    with open_dict(model.cfg.optim):
        model.cfg.optim.name = 'novograd'
        model.cfg.optim.lr = learning_rate
        model.cfg.optim.betas = [0.8, 0.5]
        model.cfg.optim.weight_decay = 0.001
        model.cfg.optim.sched.name = 'CosineAnnealing'
        model.cfg.optim.sched.warmup_steps = None
        model.cfg.optim.sched.warmup_ratio = None
        model.cfg.optim.sched.min_lr = 0.0
        model.cfg.optim.sched.last_epoch = -1

    ## AUGMENATION ##
    with open_dict(model.cfg.spec_augment):
        model.cfg.spec_augment.freq_masks = 2
        model.cfg.spec_augment.freq_width = 25
        model.cfg.spec_augment.time_masks = 2
        model.cfg.spec_augment.time_width = 0.05

        model.cfg.spec_augment.rect_freq = 50
        model.cfg.spec_augment.rect_masks = 5
        model.cfg.spec_augment.rect_time = 120

    model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

    model._wer.use_cer = False

    model._wer.log_prediction = False

def init_trainer(
        model,
        device: str,
        num_epochs: int,
        log_every_n_steps: int,
        val_every_n_epoch: int,
        name_of_run: str,
        name_of_project: str,
        model_name: str
    ):

    """
    init trainer config parameters

    Parameters:
    -----------
        model:
            model to train an

        device (str):
            device to train on: 'cpu' or 'gpu'

        num_epochs (int):
            how many epoch to train
        
        log_every_n_steps (int):
            print logs every n steps, where one step in one batch

        val_every_n_epoch (int):
            check validation metrics when training every n steps

        name_of_run (str):
            name of run for wandb logging

        name_of_project (str):
            name of project to log in

        model_name (str):
            name of model        
    
    Return:
    -------
        trainer
    """

    trainer = ptl.Trainer(devices=1,
                          accelerator=device, 
                          auto_select_gpus=True,
                          strategy=None,
                          max_epochs=num_epochs,
                          auto_lr_find=False,
                          accumulate_grad_batches=1,
                          enable_checkpointing=False,
                          logger=False,
                          log_every_n_steps=log_every_n_steps,
                          check_val_every_n_epoch=val_every_n_epoch
    )

    model.set_trainer(trainer)

    model.cfg = model._cfg

    config = exp_manager.ExpManagerConfig(
        exp_dir='experiments/',
        name=f"ASR-{model_name}-Model-ru",
        checkpoint_callback_params=exp_manager.CallbackParams(
                               monitor="val_wer",
                               mode="min",
                               always_save_nemo=True,
                               save_best_model=True),
        create_wandb_logger = True,
        wandb_logger_kwargs = {'name': name_of_run,
                               'project': name_of_project, 
                               'log_model': 'all'})

    return trainer
