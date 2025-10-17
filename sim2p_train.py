"""
Train SiM2P on paired 3D MRI-PET scans.
"""

import argparse

from sim2p import dist_util, logger
from datasets import load_data
from sim2p.resample import create_named_schedule_sampler
from sim2p.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir
)
from sim2p.train_util import TrainLoop

import torch.distributed as dist

from pathlib import Path

import wandb
import numpy as np
from glob import glob
import os

def main(args):

    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    
    dist_util.setup_dist()
    logger.configure(dir=workdir)
    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + '_resume'
        wandb.init(project="sim2p", group=args.exp, name=name, config=vars(args), mode='online' if not args.debug else 'disabled')
        logger.log("creating model and diffusion...")
    
    # save args into txt file
    arg_file = os.path.join(workdir, 'args.txt')
    with open(arg_file, 'a') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
     

    data_image_size = args.image_size
    

    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt
                if dist.get_rank() == 0:
                    logger.log('Resuming from checkpoint: ', max_ckpt)


    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    if dist.get_rank() == 0:
        wandb.watch(model, log='all')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size
        
    if dist.get_rank() == 0:
        logger.log("creating data loader...")

    data, val_data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        target_modality=args.target_modality,
        batch_size=batch_size,
        image_size=data_image_size,
        num_workers=args.num_workers,
        test_only=False,
        include_test=False,
    )
    
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=data,
        test_data=val_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        ssp_projector_path=args.ssp_projector_path,
        **sample_defaults()
    ).run_loop()


def create_argparser():
    defaults = dict(
        dims=3,
        data_dir="",
        dataset='mri2pet',
        target_modality="PET",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=10000,
        save_interval=1000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp='',
        use_fp16=True,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=2,
        use_augment=False,
        ssp_projector_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
