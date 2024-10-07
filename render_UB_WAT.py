import torch
from torch import nn
from opt import get_opts
import os
import glob
import json
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGPGv2, NGPA, NGP
from models.rendering_NGPA import render, MAX_SAMPLES
# from models.rendering import render_ori

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils.utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        # Initialize JSON results dictionary
        if not self.hparams.no_metrics:
            self.final_results = {'psnr': [], 'ssim': [], 'lpips': []}

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        if hparams.model == 'NGPGv2':
            self.model = NGPGv2(scale=self.hparams.scale, vocab_size=self.hparams.task_curr+1, rgb_act=rgb_act, dim_a = self.hparams.dim_a, dim_g = self.hparams.dim_g)
        elif hparams.model == 'NGPA':
            self.model = NGPA(scale=self.hparams.scale, vocab_size=self.hparams.task_curr+1, rgb_act=rgb_act, dim_a = self.hparams.dim_a)
        elif hparams.model == 'NGP':
            self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act)
        else:
            raise ValueError(f"Model {hparams.model} not supported!")
            
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        if split == 'test':
            poses = batch['pose']
            #directions = self.test_dataset.directions.to(self.device)
            directions = self.directions
            embed_id = batch['ts'][0].to(self.device) * torch.ones(directions.flatten().size(0), dtype=batch['ts'].dtype, device = self.device)
        else:
            raise ValueError(f"split {split} not available for this program")
        
        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]
        print(directions.device, poses.device)
        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, embed_id, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}

        self.test_dataset = dataset(split='test', **kwargs)

        # Load checkpoint first
        load_ckpt(self.model, self.hparams.weight_path)

        # Then register buffers (when set on configure_optimizers and load_ckpt is used for validation there were problems,
        # probably the buffers from the training dataset in the checkpoint get loaded)
        self.register_buffer('directions', self.test_dataset.directions.to(self.device))
        self.register_buffer('poses', self.test_dataset.poses.to(self.device))

        if not self.hparams.no_save_test:
            self.video_folder = f'results/video_demo/{self.hparams.render_fname}/{self.hparams.dataset_name}/{self.hparams.model}/{self.hparams.exp_name}_{self.hparams.render_fname}'
            os.makedirs(self.video_folder, exist_ok=True)
            self.rgb_video_writer = imageio.get_writer(self.video_folder+'/rgb.mp4', fps=60)
            self.depth_video_writer = imageio.get_writer(self.video_folder+'/depth.mp4', fps=60)

    def configure_optimizers(self):
        # define additional parameters
        # self.register_buffer('directions', self.test_dataset.directions.to(self.device))
        # self.register_buffer('poses', self.test_dataset.poses.to(self.device))

        # load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def test_dataloader(self):
        pass

    def on_train_start(self):
        pass
        
    def training_step(self, batch, batch_nb, *args):
        pass

    def on_validation_start(self):
        torch.cuda.empty_cache()
        print("start validation")
        if not self.hparams.no_metrics:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.model}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        results = self(batch, split='test')
        w, h = self.test_dataset.img_wh

        logs = {}
        if not self.hparams.no_metrics:
            # compute each metric per image
            rgb_gt = batch['rgb']   
            self.val_psnr(results['rgb'], rgb_gt)
            psnr = self.val_psnr.compute()
            logs['psnr'] = psnr
            self.val_psnr.reset()

            rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
            rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
            self.val_ssim(rgb_pred, rgb_gt)
            ssim = self.val_ssim.compute()
            logs['ssim'] = ssim
            self.val_ssim.reset()
            if self.hparams.eval_lpips:
                self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                            torch.clip(rgb_gt*2-1, -1, 1))
                lpips = self.val_lpips.compute()
                logs['lpips'] = lpips
                self.val_lpips.reset()

            # Save results for JSON
            self.final_results['psnr'].append(psnr.item())
            self.final_results['ssim'].append(ssim.item())
            if self.hparams.eval_lpips:
                self.final_results['lpips'].append(lpips.item())

            # Only save rendered test images on last epoch
            #if self.current_epoch == self.trainer.max_epochs - 1:
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            
            # Get the timestep
            timestep = batch['ts'].item()
            # Add text to RGB image
            rgb_with_text = rgb_pred.copy()
            cv2.putText(rgb_with_text, f"Timestep: {int(timestep)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Add text to depth image
            depth_with_text = depth.copy()
            cv2.putText(depth_with_text, f"Timestep: {int(timestep)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Append images to video writters
            self.rgb_video_writer.append_data(rgb_with_text)
            self.depth_video_writer.append_data(depth_with_text)
            
        return logs

    def validation_epoch_end(self, outputs):
        if not self.hparams.no_save_test: # save test image to disk
            self.rgb_video_writer.close()
            self.depth_video_writer.close()

        if not self.hparams.no_metrics: 
            psnrs = torch.stack([x['psnr'] for x in outputs])
            mean_psnr = all_gather_ddp_if_available(psnrs).mean()
            self.final_results['mean_psnr'] = np.mean(self.final_results['psnr'])
            self.log('test/psnr', mean_psnr, True)

            ssims = torch.stack([x['ssim'] for x in outputs])
            mean_ssim = all_gather_ddp_if_available(ssims).mean()
            self.final_results['mean_ssim'] = np.mean(self.final_results['ssim'])
            self.log('test/ssim', mean_ssim, True)

            if self.hparams.eval_lpips:
                lpipss = torch.stack([x['lpips'] for x in outputs])
                mean_lpips = all_gather_ddp_if_available(lpipss).mean()
                self.final_results['mean_lpips'] =  np.mean(self.final_results['lpips'])
                self.log('test/lpips_vgg', mean_lpips, True)
            
            # Save test results to JSON file
            with open(os.path.join(self.val_dir, 'test_results.json'), 'w') as f:
                json.dump(self.final_results, f)

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_nb):
        pass

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.model}/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.model}/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    if hparams.task_curr != hparams.task_number - 1:
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        check_val_every_n_epoch=hparams.num_epochs+1,
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        strategy=DDPPlugin(find_unused_parameters=False)
                                if hparams.num_gpus>1 else None,
                        num_sanity_val_steps=-1 if hparams.val_only else 0,
                        precision=16)
    else:  
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        check_val_every_n_epoch=hparams.num_epochs,
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        strategy=DDPPlugin(find_unused_parameters=False)
                                if hparams.num_gpus>1 else None,
                        num_sanity_val_steps=-1 if hparams.val_only else 0,
                        precision=16)

    trainer.validate(system, ckpt_path=hparams.ckpt_path)
