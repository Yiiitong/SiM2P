import argparse
import os

import numpy as np
import torch as th
import torchvision.utils as vutils
import torch.distributed as dist

from sim2p import dist_util, logger
from sim2p.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from sim2p.random_util import get_generator
from sim2p.karras_diffusion import karras_sample, forward_sample

from torchvision import transforms as T
import nibabel as nib
import pandas as pd
from datasets import load_data

from evaluations.ssim import *

from pathlib import Path
from tqdm import tqdm
from PIL import Image

def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir

affine = np.array([
                [1.5, 0, 0, 0],
                [0, 1.5, 0, 0],
                [0, 0, 1.5, 0],
                [0, 0, 0, 1]
            ])

def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w, l)
    """
    v_max = 1.
    # (n,)
    min_batch = min(img1.shape[0], img2.shape[0])
    img1, img2 = map(lambda t: t[:min_batch], (img1, img2))

    mse = torch.mean((img1 - img2)**2, dim=[2, 3, 4])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


@torch.no_grad()
def test():
    args = create_argparser().parse_args()

    workdir = os.path.dirname(args.model_path)

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.split("_")
    step = int(split[-1].split(".")[0])
    
    sample_dir = Path(workdir)/f'sample_{step}/w={args.guidance}_churn={args.churn_step_ratio}_step={args.steps}'
    
    dist_util.setup_dist()
    if dist.get_rank() == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    
    logger.configure(dir=workdir)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model = model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")

    test_loader = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        include_test=True,
        test_only=True,
        seed=args.seed,
        num_workers=args.num_workers,
        target_modality=args.target_modality,
    )

    args.num_samples = len(test_loader.dataset)
    
    if args.dataset == 'mri2pet':
        mae_score = []
        mse_score = []
        psnr_score = []
        ssim_score = []

        all_real = []
        all_fake = []
        
        for i, data in enumerate(test_loader):
            scan_ids = data['mri_uid']

            whole_real_scan = []
            whole_fake_scan = []

            real_B, fake_B = [], []
            
            x0_image = data['pet_scan']
            y0_image = data['mri_scan']

            x0 = x0_image.to(dist_util.dev()) * 2 - 1
            y0 = y0_image.to(dist_util.dev()) * 2 - 1

            model_kwargs = {'xT': y0}

            tabular_data = data['tabular_data'].to(dist_util.dev())
            model_kwargs['tabular'] = tabular_data

            sample, path, nfe = karras_sample(
                diffusion,
                model,
                y0,
                steps=args.steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                sigma_min=diffusion.sigma_min,
                sigma_max=diffusion.sigma_max,
                churn_step_ratio=args.churn_step_ratio,
                rho=args.rho,
                guidance=args.guidance
            )

            assert sample.shape == x0.shape, f"{sample.shape} != {x0.shape}"
            
            real_output, fake_output = x0, sample

            real_output = real_output * 0.5 + 0.5
            fake_output = fake_output * 0.5 + 0.5

            if real_output.shape[1] == 2:
                # for PETSSP
                real_output = real_output[:, 0].unsqueeze(1)
                fake_output = fake_output[:, 0].unsqueeze(1)
                assert real_output.shape[1] == 1, f"{real_output.shape}"

            real_B.append(real_output)
            fake_B.append(fake_output)

            all_real.append(real_output)
            all_fake.append(fake_output)
            
            whole_real_scan.append(real_output)
            whole_fake_scan.append(fake_output)
            whole_real_scan = torch.cat(whole_real_scan, dim=0).cpu().numpy()
            whole_fake_scan = torch.cat(whole_fake_scan, dim=0).cpu().numpy()

            if args.save_syn_scans:                       
                for batch_idx in range(whole_real_scan.shape[0]):
                    name = str(scan_ids[batch_idx])
                    # save synthetic scans
                    _whole_fake_scan = nib.Nifti1Image(whole_fake_scan[batch_idx].squeeze(), affine=affine)
                    _whole_fake_scan.to_filename(os.path.join(sample_dir, f'{name}_syn_pet.nii.gz'))

            real_B = torch.cat(real_B, dim=0)
            fake_B = torch.cat(fake_B, dim=0)

            MAE = F.l1_loss(real_B, fake_B)
            MSE = F.mse_loss(real_B, fake_B)
            PSNR = psnr(real_B, fake_B)
            SSIM = ssim3D(real_B, fake_B)   

            mae_score.append(MAE)  
            mse_score.append(MSE)  
            psnr_score.append(PSNR.mean().item())
            ssim_score.append(SSIM.item())

            
        MAE = sum(mae_score) / len(mae_score)
        MSE = sum(mse_score) / len(mse_score)
        PSNR = sum(psnr_score) / len(psnr_score)
        SSIM = sum(ssim_score) / len(ssim_score)

        print('____________________')
        print('MAE: {:.6f}'.format(MAE))
        print('MSE: {:.6f}'.format(MSE))   
        print('PSNR: {:.6f}'.format(PSNR))
        print('SSIM: {:.6f}'.format(SSIM))

        real_B = torch.cat(all_real, dim=0)
        fake_B = torch.cat(all_fake, dim=0)

        mae_err = F.l1_loss(real_B, fake_B, reduction='none')
        mse_err = F.mse_loss(real_B, fake_B, reduction='none')
        mae = mae_err.flatten(start_dim=1).mean(dim=1)
        mse = mse_err.flatten(start_dim=1).mean(dim=1)
        psnr_all = psnr_per_sample(mse, max_val=1.0)
        ssim_all = ssim3D(real_B, fake_B, size_average=False).mean(dim=1)

        df = pd.DataFrame({
            'MAE': mae.cpu().numpy(),
            'MSE': mse.cpu().numpy(),
            'PSNR': psnr_all.cpu().numpy(),
            'SSIM': ssim_all.cpu().numpy(),
        })
        df.to_csv(os.path.join(sample_dir, '_scores_per_sample.csv'), index=True)

        metrics = {
            "MAE" : (mae.mean(),  mae.std(unbiased=True)),
            "MSE" : (mse.mean(),  mse.std(unbiased=True)),
            "PSNR": (psnr_all.mean(), psnr_all.std(unbiased=True)),
            "SSIM": (ssim_all.mean(), ssim_all.std(unbiased=True)),
        }

        print('____________________')
        for metric, (mean, std) in metrics.items():
            print(f"{metric}: {mean:.6f} ± {std:.6f}")
        
        score_file = os.path.join(sample_dir, '_scores.txt')

        with open(score_file, 'a') as f:
            f.write('____________________\n')
            f.write('MAE: {:.6f}\n'.format(MAE))
            f.write('MSE: {:.6f}\n'.format(MSE))
            f.write('PSNR: {:.6f}\n'.format(PSNR))
            f.write('SSIM: {:.6f}\n'.format(SSIM))

            f.write('____________________\n')
            for metric, (mean, std) in metrics.items():
                f.write(f"{metric}: {mean:.6f} ± {std:.6f}\n")

    else:
        raise NotImplementedError

def psnr_per_sample(mse, max_val=1.0):
    """mse: tensor of shape (B,) containing per-sample MSE."""
    return 10 * torch.log10((max_val ** 2) / mse)

def create_argparser():
    defaults = dict(
        data_dir="",
        dataset='mri2pet',
        target_modality='PET',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split='test',
        churn_step_ratio=0.,
        rho=7.0,
        steps=40,
        model_path="",
        exp="",
        seed=888,
        ts="",
        upscale=False,
        num_workers=2,
        guidance=1.,
        save_syn_scans=True,
        use_fp16=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    test()