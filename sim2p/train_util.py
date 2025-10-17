import copy
import functools
import os
from pathlib import Path
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from sim2p.random_util import get_generator
from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np
from sim2p.karras_diffusion import karras_sample
from torch.utils.tensorboard import SummaryWriter

import glob 
INITIAL_LOG_LOSS_SCALE = 20.0

import wandb

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_data,
        test_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        test_interval,
        save_interval,
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=100000000,
        ssp_projector_path=None,
        **sample_kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = train_data
        self.test_data = test_data
        self.image_size = model.image_size
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.workdir = workdir
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self.writer = SummaryWriter(self.workdir + '/tb_log/')

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        num_gpus = th.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        if th.cuda.is_available() and num_gpus > 1:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

        self.generator = get_generator(sample_kwargs['generator'], self.batch_size, 888)
        self.sample_kwargs = sample_kwargs

        self.ssp_projector_path = ssp_projector_path

        if self.ssp_projector_path == '' or self.ssp_projector_path is None:
            self.ssp_projector = None
        else:
            from sim2p.ssp_projector import ProjectorNet
            # using ssp projector to include 3D-SSP as an extra supervision
            self.ssp_projector = ProjectorNet()
            self.ssp_projector.load_state_dict(th.load(self.ssp_projector_path))
            print(f"Loaded pretrained ssp projector from {self.ssp_projector_path}")
    

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log('Resume step: ', self.resume_step)
                
            self.model.load_state_dict(
                # dist_util.load_state_dict(
                #     resume_checkpoint, map_location=dist_util.dev()
                # ),
                th.load(resume_checkpoint, map_location=dist_util.dev()),
            )
        
            dist.barrier()

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            # state_dict = dist_util.load_state_dict(
            #     ema_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

            dist.barrier()
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint.split('/')[-1].startswith("latest"):
            prefix = 'latest_'
        else:
            prefix = ''
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"{prefix}opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            # state_dict = dist_util.load_state_dict(
            #     opt_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
            dist.barrier()

    def preprocess(self, x):
        # rescale to [-1, 1]
        x =  x * 2 - 1              
        return x

    def run_loop(self):
        while True:
            best_val_loss = 999999.0

            for batch_data in self.data:
                batch = batch_data['pet_scan']
                cond = batch_data['mri_scan']
                if 'ssp_scan' in batch_data:
                    ssp_scan = batch_data['ssp_scan']
                else:
                    ssp_scan = None
                
                tabular_data = batch_data['tabular_data']
                
                if not (not self.lr_anneal_steps or self.step < self.total_training_steps):
                    # Save the last checkpoint if it wasn't already saved.
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return
                
                # scale to [-1, 1]  
                batch = self.preprocess(batch)
                
                if isinstance(cond, th.Tensor) and batch.ndim == cond.ndim:
                    xT = self.preprocess(cond)
                    cond = {'xT': xT}
                else:
                    cond['xT'] = self.preprocess(cond['xT'])
                
                cond['tabular'] = tabular_data

                if ssp_scan is not None:
                    ssp_scan = self.preprocess(ssp_scan)
                    cond['ssp_scan'] = ssp_scan

                took_step = self.run_step(batch, cond)
                if took_step and self.step % self.log_interval == 0:
                    logs = logger.dumpkvs()

                    log_items = list(logs.items())  # Create a list copy of items

                    for key, value in log_items:
                        self.writer.add_scalar(f'logs/{key}', value, global_step=logs['step'])

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)
                        
                if took_step and self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                    
                    test_batch_data = next(iter(self.test_data))
                    test_batch = test_batch_data['pet_scan']
                    test_cond = test_batch_data['mri_scan']
                    if 'ssp_scan' in test_batch_data:
                        test_ssp_scan = test_batch_data['ssp_scan']
                    else:
                        test_ssp_scan = None
                    tabular_data = test_batch_data['tabular_data']
                    
                    test_batch = self.preprocess(test_batch)
                    if isinstance(test_cond, th.Tensor) and test_batch.ndim == test_cond.ndim:
                        test_cond = {'xT': self.preprocess(test_cond)}
                    else:
                        test_cond['xT'] = self.preprocess(test_cond['xT'])
                    
                    test_cond['tabular'] = tabular_data

                    if test_ssp_scan is not None:
                        test_ssp_scan = self.preprocess(test_ssp_scan)
                        test_cond['ssp_scan'] = test_ssp_scan
              
                    self.run_test_step(test_batch, test_cond)
                    logs = logger.dumpkvs()

                    log_items = list(logs.items())

                    if logs['test_loss'] < best_val_loss:
                        best_val_loss = logs['test_loss']
                        self.save_best_val()


                    global_step = logs.get('step', self.step)

                    for key, value in log_items:
                        self.writer.add_scalar(f'logs_val/{key}', value, global_step=global_step)

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)
                

                if took_step and self.step % self.save_interval_for_preemption == 0:
                    self.save(for_preemption=True)
        

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        # print(f"Current GPU memory allocated: {th.cuda.memory_allocated(th.device('cuda'))/1024**3:.2f} GB")
        # print(f"Current GPU memory reserved: {th.cuda.memory_reserved(th.device('cuda'))/1024**3:.2f} GB")
        return took_step

    def run_test_step(self, batch, cond):
        with th.no_grad():
            self.forward_backward(batch, cond, train=False)

    def forward_backward(self, batch, cond, train=True):
        if train:
            self.mp_trainer.zero_grad()
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                    self.diffusion.training_bridge_losses,
                    self.ddp_model,
                    micro,
                    t,
                    self.ssp_projector,
                    model_kwargs=micro_cond,
                )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler) and train:
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k if train else 'test_'+k: v * weights for k, v in losses.items()}
            )
            if train:
                self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f'{(self.step):010d}')[0]+'*'
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 0:
                earliest = min(freq_states, key=lambda x: x.split('_')[-1].split('.')[0])
                os.remove(earliest)
                    

        # if dist.get_rank() == 0 and for_preemption:
        #     maybe_delete_earliest(get_blob_logdir())
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_{(self.step):010d}.pt"
                    maybe_delete_earliest(filename)
                else:
                    filename = f"ema_{rate}_{(self.step):010d}.pt"
                    maybe_delete_earliest(filename)
                if for_preemption:
                    filename = f"freq_{filename}"
                    maybe_delete_earliest(filename)
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            filename = f"opt_{(self.step):010d}.pt"
            maybe_delete_earliest(filename)
            if for_preemption:
                filename = f"freq_{filename}"
                maybe_delete_earliest(filename)
                
            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()
    
    
    def save_best_val(self):
        """
        Saves the EMA model checkpoint with prefix 'best_val_' and deletes the previous one.
        """

        # Function to remove previous best checkpoints
        def delete_previous_best(rate):
            # Construct the pattern for previous best_val checkpoints
            wc = os.path.join(get_blob_logdir(), f"best_val_ema_{rate}_*.pt")
            previous_ckpts = glob.glob(wc)

            # If any previous checkpoint exists, delete the earliest one
            if previous_ckpts:
                earliest_ckpt = min(previous_ckpts, key=os.path.getctime)  # Get the oldest checkpoint
                os.remove(earliest_ckpt)
                print(f"Deleted previous best_val checkpoint: {earliest_ckpt}")

        # save the new best EMA checkpoint
        def save_best_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"Saving best_val EMA model {rate}...")

                # Define checkpoint filename with step number
                filename = f"best_val_ema_{rate}_{(self.step):010d}.pt"
                ckpt_path = os.path.join(get_blob_logdir(), filename)

                # Delete previous best checkpoint
                delete_previous_best(rate)

                # Save the new checkpoint
                with bf.BlobFile(ckpt_path, "wb") as f:
                    th.save(state_dict, f)
                print(f"Best validation checkpoint saved at: {ckpt_path}")

        # Save best EMA models
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_best_checkpoint(rate, params)

        # Ensure all processes are synchronized
        dist.barrier()



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    # if dist.get_rank() == 0:
    #     logger.log("looking for latest checkpoint...")
    # checkpoints = sorted(
    #     glob.glob(os.path.join(get_blob_logdir(), "model_*.pt")),
    #     key=lambda x: parse_resume_step_from_filename(x),
    # )
    # if not checkpoints:
    #     return None
    # return checkpoints[-1]
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split('/')[-1].startswith("latest"):
        prefix = 'latest_'
    else:
        prefix = ''
    filename = f"{prefix}ema_{rate}_{(step):010d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
