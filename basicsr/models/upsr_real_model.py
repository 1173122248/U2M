import torch
import torch.amp as amp
import torch.nn as nn
import functools
import math
import lpips
import os
import os.path as osp
import pyiqa
import numpy as np
import random
import tqdm
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import matplotlib.pyplot as plt


# from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_obj_from_str, get_root_logger, ImageSpliterTh, imwrite, tensor2img
from basicsr.archs import build_network
from basicsr.archs.uncertainty_net_arch import (
    UncertaintyPredictionNet, 
    AdaptiveNoiseScheduler, 
    UncertaintyLoss,
    compute_edge_map,
    compute_texture_complexity
)
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from .base_model import BaseModel
from .sr_model import SRModel
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from contextlib import nullcontext
from copy import deepcopy
from torch.nn.parallel import DataParallel
from collections import OrderedDict
from torchvision import transforms
from PIL import Image


class UPSRRealModel(SRModel):
    """Diffusion SR model for single image super-resolution."""

    def __init__(self, opt):
        # === 关键：在super().__init__()之前初始化这些属性 ===
        self.opt = opt
        self.use_uncertainty_net = opt.get('use_uncertainty_net', False)
        # 新增：U²M模块标志
        self.use_u2m = opt.get('use_u2m', False)
        # 初始化decay_scheduler为None(在super()之前,防止init_training_settings访问时出错)
        self.decay_scheduler = None
        
        super(UPSRRealModel, self).__init__(opt)

        logger = get_root_logger()

        self.sf = self.opt['scale']

        # define network net_mse g(\cdot)
        net_mse_opt = self.opt['network_mse']
        assert net_mse_opt['ckpt']['path'] is not None, 'ckpt_path is required for net_mse'
        logger.info(f"Restoring network_mse from {net_mse_opt['ckpt']['path']}")

        self.net_mse = build_network(net_mse_opt)
        param_key = net_mse_opt['ckpt'].get('param_key_mse', 'params_ema')
        self.load_network(self.net_mse, net_mse_opt['ckpt']['path'], net_mse_opt['ckpt'].get('strict_load_mse', True), param_key)
        self.net_mse.eval()
        for name, param in self.net_mse.named_parameters():
            param.requires_grad = False
        self.net_mse = self.net_mse.to(self.device)

        # define base_diffusion
        diff_opt = self.opt['diffusion']
        self.base_diffusion = build_network(diff_opt)
        
        # === 可学习时间衰减调度器实例化 ===
        # (self.decay_scheduler已在super()之前初始化为None)
        if self.use_u2m and diff_opt.get('use_dynamic_u2m', False):
            if diff_opt.get('learnable_decay', False):
                from models.learnable_decay_scheduler import create_decay_scheduler
                scheduler_type = diff_opt.get('decay_scheduler_type', 'parametric')
                init_strategy = diff_opt.get('decay_init_strategy', 'linear')
                self.decay_scheduler = create_decay_scheduler(
                    scheduler_type=scheduler_type,
                    init_strategy=init_strategy
                ).to(self.device)
                logger.info(f"[Learnable Decay Scheduler] Created: type={scheduler_type}, init={init_strategy}")
        
        # === 推理时加载U²M网络 ===
        if self.use_u2m and not self.is_train:
            from basicsr.archs.u2m_arch import U2M, TimeDecayScheduler
            
            logger.info("=== Loading U2M for Inference ===")
            
            # 初始化U²M网络
            u2m_opt = self.opt.get('network_u2m', {})
            self.net_u2m = U2M(
                in_channels=u2m_opt.get('in_channels', 6),
                base_channels=u2m_opt.get('base_channels', 32),
                num_encoder_blocks=u2m_opt.get('num_encoder_blocks', 4),
                use_attention=u2m_opt.get('use_attention', True)
            )
            self.net_u2m = self.model_to_device(self.net_u2m)
            self.net_u2m.eval()
            
            # 时间衰减调度器
            self.time_decay_scheduler = TimeDecayScheduler(
                decay_type=u2m_opt.get('decay_type', 'cosine'),
                min_value=u2m_opt.get('decay_min_value', 0.0)
            )
            self.time_decay_scheduler = self.model_to_device(self.time_decay_scheduler)
            
            # 加载预训练的U²M权重
            load_u2m_path = self.opt['path'].get('pretrain_network_u2m', None)
            if load_u2m_path is not None:
                logger.info(f"Loading U2M from: {load_u2m_path}")
                self.load_network(self.net_u2m, load_u2m_path, 
                                self.opt['path'].get('strict_load_u2m', True), 
                                'u2m')  # 使用u2m作为param_key(与save_network保持一致)
                logger.info(f"U2M loaded successfully - Params: {sum(p.numel() for p in self.net_u2m.parameters()) / 1e6:.2f}M")
            else:
                logger.warning("No pretrain_network_u2m specified for inference!")
        
        # === 不确定性网络将在 init_training_settings() 中初始化 ===
        # 不要在这里设置为 None，会覆盖 init_training_settings() 中的初始化！
        
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 160)

        # define lpips loss
        loss_lpips = self.metric_lpips = pyiqa.create_metric('lpips-vgg', as_loss=True, device=self.device)
        self.loss_lpips = loss_lpips

        if self.opt['rank'] == 0:
            self.metrics_fr = {}
            self.metrics_nr = {}

            for metric_name, metric_opt in self.opt['val']['metrics'].items():
                # Skip FID metric - it requires folder paths for comparison
                if metric_name.lower() == 'fid':
                    logger.warning(f"FID metric is not supported in online validation (requires folder paths). Skipping.")
                    continue
                
                # Skip PI metric - it will be calculated from NIQE only
                if metric_name.lower() == 'pi':
                    continue
                
                # Skip MA metric - removed due to compatibility issues
                if metric_name.lower() == 'ma':
                    logger.warning(f"MA metric skipped (use MUSIQ instead for similar functionality)")
                    continue
                    
                if metric_opt.get('fr', True):
                    self.metrics_fr[metric_name] = pyiqa.create_metric(metric_name, device=self.device)
                else:
                    self.metrics_nr[metric_name] = pyiqa.create_metric(metric_name, device=self.device)
            
            # If PI is requested, ensure NIQE is available (simplified PI = NIQE)
            if 'pi' in self.opt['val']['metrics']:
                if 'niqe' not in self.metrics_nr:
                    self.metrics_nr['niqe'] = pyiqa.create_metric('niqe', device=self.device)

            self.metrics_fr['psnr'] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=self.device)
            self.metrics_fr['ssim'] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device=self.device)

        # === Early Stopping 初始化 ===
        self.early_stopping_enabled = opt.get('train', {}).get('early_stopping', {}).get('enabled', False)
        if self.early_stopping_enabled:
            self.early_stopping_metric = opt['train']['early_stopping'].get('metric', 'clipiqa')
            self.early_stopping_patience = opt['train']['early_stopping'].get('patience', 3)
            self.early_stopping_min_delta = opt['train']['early_stopping'].get('min_delta', 0.001)
            self.early_stopping_mode = opt['train']['early_stopping'].get('mode', 'max')  # max or min
            
            # 记录未改进次数
            self.early_stopping_counter = 0
            self.early_stopping_best_value = None
            self.early_stopping_best_iter = 0
            
            logger.info(f"=== Early Stopping Enabled ===")
            logger.info(f"  Metric: {self.early_stopping_metric}")
            logger.info(f"  Patience: {self.early_stopping_patience} (x2000 iters)")
            logger.info(f"  Min Delta: {self.early_stopping_min_delta}")
            logger.info(f"  Mode: {self.early_stopping_mode}")


    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def init_training_settings(self):

        logger = get_root_logger()
        logger.info(">>> UPSRRealModel.init_training_settings() called <<<")
        logger.info(f">>> use_uncertainty_net = {self.use_uncertainty_net} <<<")
        logger.info(f">>> use_u2m = {self.use_u2m} <<<")
        
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = True
            self.perceptual_weight = train_opt['perceptual_opt']['lpips_weight']
        else:
            self.cri_perceptual = False

        # === 可学习衰减调度器的优化器 ===
        if self.decay_scheduler is not None:
            optim_params = list(self.decay_scheduler.parameters())
            self.optimizer_decay = torch.optim.Adam(optim_params, lr=1e-4, weight_decay=0.0)
            self.optimizers.append(self.optimizer_decay)
            logger.info(f"[Learnable Decay Scheduler] Optimizer created with lr=1e-4")
        
        # === 新增：U²M模块的训练设置 ===
        if self.use_u2m:
            from basicsr.archs.u2m_arch import U2M, TimeDecayScheduler
            
            logger.info("=== Initializing U2M Module (START) ===")
            
            # 训练策略标志
            self.joint_training = train_opt.get('joint_u2m_training', True)
            self.freeze_main_network = train_opt.get('freeze_main_network', False)
            
            # 初始化U²M网络
            u2m_opt = self.opt.get('network_u2m', {})
            logger.info(f"U2M config: {u2m_opt}")
            
            self.net_u2m = U2M(
                in_channels=u2m_opt.get('in_channels', 6),
                base_channels=u2m_opt.get('base_channels', 32),
                num_encoder_blocks=u2m_opt.get('num_encoder_blocks', 4),
                use_attention=u2m_opt.get('use_attention', True)
            )
            self.net_u2m = self.model_to_device(self.net_u2m)
            self.net_u2m.train()
            
            logger.info(f"U2M - Params: {sum(p.numel() for p in self.net_u2m.parameters()) / 1e6:.2f}M")
            
            # 时间衰减调度器（用于反向去噪阶段）
            self.time_decay_scheduler = TimeDecayScheduler(
                decay_type=u2m_opt.get('decay_type', 'cosine'),
                min_value=u2m_opt.get('decay_min_value', 0.0)
            )
            self.time_decay_scheduler = self.model_to_device(self.time_decay_scheduler)
            
            # 加载预训练的U²M权重（如果有）
            load_u2m_path = self.opt['path'].get('pretrain_network_u2m', None)
            if load_u2m_path is not None:
                logger.info(f"Loading pretrained U2M from: {load_u2m_path}")
                self.load_network(self.net_u2m, load_u2m_path, 
                                self.opt['path'].get('strict_load_u2m', True), 
                                'u2m')
                logger.info("[OK] Loaded U2M weights")
            
            # === 初始化新的U2MLoss ===
            from basicsr.losses.u2m_loss import U2MLoss
            
            u2m_loss_opt = train_opt.get('u2m_loss_opt', {})
            
            self.u2m_loss = U2MLoss(
                lambda_r=u2m_loss_opt.get('lambda_r', 1.0),       # residual alignment
                lambda_c=u2m_loss_opt.get('lambda_c', 0.5),       # calibration
                lambda_s=u2m_loss_opt.get('lambda_s', 0.01),      # smoothness
                lambda_m=u2m_loss_opt.get('lambda_m', 0.01),      # magnitude
                sigma_min=u2m_loss_opt.get('sigma_min', 0.01),
                sigma_max=u2m_loss_opt.get('sigma_max', 0.45),
                calib_samples=u2m_loss_opt.get('calib_samples', 1024),
                gamma=u2m_loss_opt.get('gamma', 1.0)
            )
            self.u2m_loss = self.model_to_device(self.u2m_loss)
            
            logger.info(f"U2M Loss:")
            logger.info(f"  lambda_r (res_align): {u2m_loss_opt.get('lambda_r', 1.0)}")
            logger.info(f"  lambda_c (calib): {u2m_loss_opt.get('lambda_c', 0.5)}")
            logger.info(f"  lambda_s (smooth): {u2m_loss_opt.get('lambda_s', 0.01)}")
            logger.info(f"  lambda_m (mag): {u2m_loss_opt.get('lambda_m', 0.01)}")
            logger.info(f"  sigma_min: {u2m_loss_opt.get('sigma_min', 0.01)}")
            logger.info(f"  sigma_max: {u2m_loss_opt.get('sigma_max', 0.45)}")
            
            if not self.joint_training:
                logger.info("U2M will be trained separately (no main network updates)")
            if self.freeze_main_network:
                logger.info("Freezing main network (only U2M trainable)")
                for param in self.net_g.parameters():
                    param.requires_grad = False
                if hasattr(self, 'net_mse') and self.net_mse is not None:
                    for param in self.net_mse.parameters():
                        param.requires_grad = False
            
            logger.info("=== U2M Module Initialized (SUCCESS) ===")
        
        # === 原有：不确定性网络的训练设置 ===
        elif self.use_uncertainty_net:
            logger = get_root_logger()
            logger.info("=== Initializing Uncertainty Network (START) ===")
            
            # 先设置训练策略标志（在创建网络之前）
            self.joint_training = train_opt.get('joint_uncertainty_training', True)
            self.freeze_main_network = train_opt.get('freeze_main_network', False)
            
            # 初始化不确定性预测网络
            logger.info("Creating Uncertainty Prediction Network...")
            uncertainty_opt = self.opt.get('network_uncertainty', {})
            logger.info(f"Uncertainty config: {uncertainty_opt}")
            
            self.net_uncertainty = UncertaintyPredictionNet(
                base_channels=uncertainty_opt.get('base_channels', 32),
                num_blocks=uncertainty_opt.get('num_blocks', 4)
            )
            # 使用 model_to_device 来正确处理多GPU情况
            self.net_uncertainty = self.model_to_device(self.net_uncertainty)
            
            self.noise_scheduler = AdaptiveNoiseScheduler(
                min_noise=uncertainty_opt.get('min_noise', 0.2),
                max_noise=uncertainty_opt.get('max_noise', 1.0),
                sharpness=uncertainty_opt.get('sharpness', 2.0)
            )
            # noise_scheduler 也需要正确处理多GPU
            self.noise_scheduler = self.model_to_device(self.noise_scheduler)
            
            self.net_uncertainty.train()
            self.noise_scheduler.train()
            
            logger.info(f"Uncertainty Net - Params: {sum(p.numel() for p in self.net_uncertainty.parameters()) / 1e6:.2f}M")
            
            # 加载预训练的不确定性网络权重（如果有）
            load_uncertainty_path = self.opt['path'].get('pretrain_network_uncertainty', None)
            if load_uncertainty_path is not None:
                logger.info(f"Loading pretrained uncertainty network from: {load_uncertainty_path}")
                load_net = torch.load(load_uncertainty_path, map_location=lambda storage, loc: storage)
                
                # 加载不确定性网络
                if 'uncertainty_net' in load_net:
                    self.load_network(self.net_uncertainty, load_uncertainty_path, 
                                    self.opt['path'].get('strict_load_uncertainty', True), 
                                    'uncertainty_net')
                    logger.info("✓ Loaded uncertainty network weights")
                
                # 加载噪声调度器
                if 'noise_scheduler' in load_net:
                    self.load_network(self.noise_scheduler, load_uncertainty_path, 
                                    self.opt['path'].get('strict_load_uncertainty', True), 
                                    'noise_scheduler')
                    logger.info("✓ Loaded noise scheduler weights")
            
            logger.info("=== Uncertainty Network Initialized (SUCCESS) ===")
            
            # 不确定性损失函数
            uncertainty_loss_opt = train_opt.get('uncertainty_loss_opt', {})
            self.uncertainty_loss = UncertaintyLoss(
                lambda_self_sup=uncertainty_loss_opt.get('lambda_self_sup', 1.0),
                lambda_edge=uncertainty_loss_opt.get('lambda_edge', 0.3),
                lambda_texture=uncertainty_loss_opt.get('lambda_texture', 0.3),
                lambda_smooth=uncertainty_loss_opt.get('lambda_smooth', 0.1),
                lambda_range=uncertainty_loss_opt.get('lambda_range', 0.2)
            )
            
            if not self.joint_training:
                logger.info("Uncertainty network will be trained separately (no main network updates)")
            if self.freeze_main_network:
                logger.info("Freezing main network parameters (only uncertainty network trainable)")
                # 冻结主干网络
                for param in self.net_g.parameters():
                    param.requires_grad = False
                # 冻结MSE网络（如果存在）
                if hasattr(self, 'net_mse') and self.net_mse is not None:
                    for param in self.net_mse.parameters():
                        param.requires_grad = False
        else:
            # 如果不使用不确定性网络，也要初始化这些属性
            self.joint_training = False
            self.freeze_main_network = False

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.amp_scaler = amp.GradScaler() if self.opt['train'].get('use_fp16', False) else None

    def setup_optimizers(self):
        """重写优化器设置，支持U²M和不确定性网络"""
        train_opt = self.opt['train']
        
        # 主网络优化器（如果不冻结的话）
        if not self.freeze_main_network:
            optim_params = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')

            optim_type = train_opt['optim_g'].pop('type')
            self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
            self.optimizers.append(self.optimizer_g)
        else:
            logger = get_root_logger()
            logger.info("Main network frozen - no optimizer created for net_g")
            self.optimizer_g = None
        
        # U²M优化器
        if self.use_u2m:
            logger = get_root_logger()
            logger.info("Setting up optimizer for U2M")
            
            u2m_optim_opt = train_opt.get('optim_u2m', train_opt.get('optim_g', {}).copy())
            u2m_params = list(self.net_u2m.parameters())
            
            optim_type_u2m = u2m_optim_opt.pop('type', 'AdamW')
            self.optimizer_u2m = self.get_optimizer(
                optim_type_u2m, 
                u2m_params, 
                **u2m_optim_opt
            )
            self.optimizers.append(self.optimizer_u2m)
        
        # 不确定性网络优化器（旧版本，保持兼容）
        elif self.use_uncertainty_net:
            logger = get_root_logger()
            logger.info("Setting up optimizer for uncertainty network")
            
            uncertainty_optim_opt = train_opt.get('optim_uncertainty', train_opt['optim_g'].copy())
            uncertainty_params = list(self.net_uncertainty.parameters()) + \
                               list(self.noise_scheduler.parameters())
            
            optim_type_uncertainty = uncertainty_optim_opt.pop('type', 'AdamW')
            self.optimizer_uncertainty = self.get_optimizer(
                optim_type_uncertainty, 
                uncertainty_params, 
                **uncertainty_optim_opt
            )
            self.optimizers.append(self.optimizer_uncertainty)

    def backward_step(self, dif_loss_wrapper, micro_lq, micro_gt, num_grad_accumulate, tt):
        loss_dict = OrderedDict()

        context = amp.autocast if self.opt['train'].get('use_fp16', False) else nullcontext
        with context(device_type="cuda"):
            losses, x_t, x0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            l_pix = losses['loss'].mean() / num_grad_accumulate
            
            l_total = l_pix
            loss_dict['l_pix'] = l_pix 

            if self.cri_perceptual:
                # 计算per-sample LPIPS
                l_lpips_raw = self.loss_lpips(x0_pred.clamp(-1., 1.), micro_gt).to(x0_pred.dtype).view(-1)
                if torch.any(torch.isnan(l_lpips_raw)):
                    l_lpips_raw = torch.nan_to_num(l_lpips_raw, nan=0.0)
                
                l_lpips = l_lpips_raw.mean() / num_grad_accumulate * self.perceptual_weight

                l_total += l_lpips
                loss_dict['l_lpips'] = l_lpips 

        if self.amp_scaler is None:
            l_total.backward()
        else:
            self.amp_scaler.scale(l_total).backward()

        return loss_dict, x_t, x0_pred

    def optimize_parameters(self, current_iter):
        # DEBUG: 检查不确定性网络状态
        if current_iter == 1:
            logger = get_root_logger()
            logger.info(f"[DEBUG] optimize_parameters - use_u2m: {self.use_u2m}")
            if self.use_u2m and hasattr(self, 'net_u2m'):
                logger.info(f"[DEBUG] optimize_parameters - net_u2m type: {type(self.net_u2m)}")
                logger.info(f"[DEBUG] optimize_parameters - net_u2m is None: {self.net_u2m is None}")
            if hasattr(self, 'net_g'):
                logger.info(f"[DEBUG] optimize_parameters - net_g type: {type(self.net_g)}")
        
        current_batchsize = self.lq.shape[0]
        micro_batchsize = self.opt['datasets']['train']['micro_batchsize']
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        # 只有在主网络未冻结时才清零梯度
        if self.optimizer_g is not None:
            self.optimizer_g.zero_grad()

        loss_dict = OrderedDict()
        loss_dict['l_pix'] = 0
        if self.cri_perceptual:
            loss_dict['l_lpips'] = 0

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_lq = self.lq[jj:jj+micro_batchsize,]
            micro_gt = self.gt[jj:jj+micro_batchsize,]


            last_batch = (jj+micro_batchsize >= current_batchsize)
            if self.opt['diffusion'].get('one_step', False):
                tt = torch.ones(
                    size=(micro_gt.shape[0],),
                    device=self.lq.device,
                    dtype=torch.int32,
                    ) * (self.base_diffusion.num_timesteps - 1)
            else:
                tt = torch.randint(
                        0, self.base_diffusion.num_timesteps,
                        size=(micro_gt.shape[0],),
                        device=self.lq.device,
                        )
            
            with torch.no_grad():        
                # y_0
                micro_lq_bicubic = torch.nn.functional.interpolate(
                        micro_lq, scale_factor=self.sf, mode='bicubic', align_corners=False,
                        )
                # g(y_0)
                micro_sr_mse = (self.net_mse(micro_lq * 0.5 + 0.5) - 0.5) / 0.5

            # === 正向加噪阶段的不确定性预测 ===
            if self.use_u2m:
                # 使用U²M模块：输入 [LR_upsampled, SR_pre]
                # 注意：需要将LR上采样到HR尺寸以匹配SR_MSE
                micro_lq_up = torch.nn.functional.interpolate(
                    micro_lq, size=micro_sr_mse.shape[2:], 
                    mode='bilinear', align_corners=False
                )
                u2m_input_fwd = torch.cat([micro_lq_up, micro_sr_mse], dim=1)  # (B, 6, H, W)
                
                # 关键修正: 主干训练阶段,U²M只提供不确定性指导,不接收梯度
                # U²M的梯度更新仅在_train_u2m阶段进行(包括联合训练的主干回传)
                with torch.no_grad():
                    sigma_fwd, global_v_fwd = self.net_u2m(u2m_input_fwd)  # (B, 1, H, W), (B, 1, 1, 1)
                # 扩展到3通道以匹配扩散过程
                micro_uncertainty = sigma_fwd.expand_as(micro_sr_mse)
            elif self.use_uncertainty_net:
                # 使用可训练的不确定性网络（旧版本）
                raw_uncertainty = self.net_uncertainty(
                    micro_lq, micro_sr_mse, micro_lq_bicubic, 
                    return_features=False
                )
                # 应用自适应噪声调度器
                micro_uncertainty = self.noise_scheduler(raw_uncertainty, smooth=True)
                # 扩展单通道不确定性图到3通道以匹配扩散过程
                micro_uncertainty = micro_uncertainty.expand_as(micro_sr_mse)
            else:
                # 使用原始固定计算方式
                with torch.no_grad():
                    if self.opt['diffusion']['un'] > 0:
                        diff = (micro_sr_mse - micro_lq_bicubic) / 2
                        un_max = self.opt['diffusion']['un']
                        b_un = self.opt['diffusion']['min_noise']
                        micro_uncertainty = torch.abs(diff).clamp_(0., un_max) / un_max
                        micro_uncertainty = b_un + (1 - b_un) * micro_uncertainty
                    else:
                        micro_uncertainty = torch.ones_like(micro_sr_mse)

            # n
            noise = torch.randn_like(micro_sr_mse)

            lq_cond = nn.PixelUnshuffle(self.sf)(torch.cat([micro_sr_mse, micro_lq_bicubic], dim=1))


            model_kwargs={'lq':lq_cond,} if self.opt['network_g']['params']['cond_lq'] else None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.net_g,
                micro_gt,
                micro_lq_bicubic,
                micro_sr_mse,
                micro_uncertainty,
                tt,
                model_kwargs=model_kwargs,
                noise=noise,
            )

            if last_batch or self.opt['num_gpu'] <= 1:
                losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt)
            else:
                # no_sync() 只在 DistributedDataParallel 中可用
                if self.opt.get('dist', False):
                    with self.net_g.no_sync():
                        losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt)
                else:
                    # DataParallel 不需要 no_sync
                    losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt)
            
            loss_dict['l_pix'] += losses['l_pix']
            if self.cri_perceptual:
                loss_dict['l_lpips'] += losses['l_lpips']
        
        # === 更新主网络 ===
        if not self.freeze_main_network:
            if self.opt['train'].get('use_fp16', False):
                self.amp_scaler.step(self.optimizer_g)
                self.amp_scaler.update()
            else:
                self.optimizer_g.step()
            self.net_g.zero_grad()
        
        # === U²M训练 ===
        if self.use_u2m:
            self._train_u2m(current_iter, loss_dict)
        
        # === 不确定性网络训练（旧版本） ===
        elif self.use_uncertainty_net:
            self._train_uncertainty_network(current_iter, loss_dict)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def _train_u2m(self, current_iter, loss_dict):
        """训练U²M模块 - 联合训练策略
        
        联合训练机制:
        1. U²M损失反向传播到主干网络,引导主干产生与不确定性一致的输出
        2. U²M同时优化自身预测能力和主干重建质量
        3. 两个模块通过共享梯度相互促进,避免脱节
        """
        # 联合训练配置
        joint_training = self.opt['train'].get('joint_u2m_training', True)
        u2m_to_backbone_weight = self.opt['train'].get('u2m_to_backbone_weight', 0.1)
        
        self.optimizer_u2m.zero_grad()
        if joint_training:
            # 联合训练时也需要清零主干梯度,因为U²M损失会传播到主干
            self.optimizer_g.zero_grad()
        
        current_batchsize = self.lq.shape[0]
        micro_batchsize = self.opt['datasets']['train']['micro_batchsize']
        
        # 新的loss记录
        u2m_res_align_total = 0.0
        u2m_calib_total = 0.0
        u2m_smooth_total = 0.0
        u2m_mag_total = 0.0
        u2m_loss_total = 0.0
        
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_lq = self.lq[jj:jj+micro_batchsize,]
            micro_gt = self.gt[jj:jj+micro_batchsize,]
            
            # === 关键修改: 移除torch.no_grad(),允许梯度回传到主干 ===
            micro_lq_bicubic = torch.nn.functional.interpolate(
                micro_lq, scale_factor=self.sf, mode='bicubic', align_corners=False
            )
            
            if joint_training:
                # 联合训练: 主干SR保留梯度,U²M损失可以回传
                micro_sr_mse = (self.net_mse(micro_lq * 0.5 + 0.5) - 0.5) / 0.5
            else:
                # 独立训练: 主干SR不参与U²M训练
                with torch.no_grad():
                    micro_sr_mse = (self.net_mse(micro_lq * 0.5 + 0.5) - 0.5) / 0.5
            
            # U²M前向传播：输入 [LR_upsampled, SR_pre]
            micro_lq_up = torch.nn.functional.interpolate(
                micro_lq, size=micro_sr_mse.shape[2:],
                mode='bilinear', align_corners=False
            )
            u2m_input = torch.cat([micro_lq_up, micro_sr_mse], dim=1)
            sigma_pred, global_v_train = self.net_u2m(u2m_input)  # (B, 1, H, W), (B, 1, 1, 1)
            
            # === 使用新的U2MLoss ===
            # pred=SR_MSE, gt=GT, lr=LR_bicubic, sigma=σ预测
            batch_loss, batch_losses = self.u2m_loss(
                sigma=sigma_pred,
                pred=micro_sr_mse,
                gt=micro_gt,
                lr=micro_lq_bicubic
            )
            
            # 累积各项损失
            u2m_loss_total += batch_loss
            u2m_res_align_total += batch_losses['res_align']  # 原始分项损失用于监控
            u2m_calib_total += batch_losses['calib']
            u2m_smooth_total += batch_losses['smooth']
            u2m_mag_total += batch_losses['mag']
            
            # DEBUG: 第一次迭代打印调试信息
            if current_iter == 1 and jj == 0:
                logger = get_root_logger()
                logger.info(f"[DEBUG] U2M Loss:")
                logger.info(f"  sigma_pred range: [{sigma_pred.min().item():.6f}, {sigma_pred.max().item():.6f}]")
                logger.info(f"  sigma_pred mean: {sigma_pred.mean().item():.6f}")
                logger.info(f"  sigma_pred std: {sigma_pred.std().item():.6f}")
                logger.info(f"  res_align: {batch_losses['res_align'].item():.6f}")
                logger.info(f"  calib: {batch_losses['calib'].item():.6f}")
                logger.info(f"  smooth: {batch_losses['smooth'].item():.6f}")
                logger.info(f"  mag: {batch_losses['mag'].item():.6f}")
                logger.info(f"  total: {batch_loss.item():.6f}")
        
        # 反向传播
        if joint_training:
            # 联合训练: U²M损失乘以权重后回传到主干
            (u2m_loss_total * u2m_to_backbone_weight).backward()
            # 更新两个优化器
            self.optimizer_u2m.step()
            self.optimizer_g.step()
            
            if current_iter == 1:
                logger = get_root_logger()
                logger.info(f"[Joint Training] U2M loss will propagate to backbone with weight={u2m_to_backbone_weight}")
        else:
            # 独立训练: U²M损失只更新U²M网络
            u2m_loss_total.backward()
            self.optimizer_u2m.step()
        
        # 记录损失
        loss_dict['l_u2m'] = u2m_loss_total
        loss_dict['l_u2m_res_align'] = u2m_res_align_total
        loss_dict['l_u2m_calib'] = u2m_calib_total
        loss_dict['l_u2m_smooth'] = u2m_smooth_total
        loss_dict['l_u2m_mag'] = u2m_mag_total
    
    def _backup_old_train_u2m(self, current_iter, loss_dict):
        """旧版训练代码(备份,已弃用)"""
        # 此函数已被新的_train_u2m替换
        # === 损失2：Heteroscedastic Loss（如果有GT）===
        if False:  # 已禁用
            if self.use_heteroscedastic:
                # 简单前向推理获取预测
                with torch.no_grad():
                    sigma_3ch = sigma_pred.expand_as(micro_sr_mse)
                    tt_sample = torch.randint(
                        0, self.base_diffusion.num_timesteps // 2,  # 使用前半段时间步
                        size=(micro_gt.shape[0],),
                        device=self.lq.device
                    )
                    
                    lq_cond = nn.PixelUnshuffle(self.sf)(
                        torch.cat([micro_sr_mse, micro_lq_bicubic], dim=1)
                    )
                    model_kwargs = {'lq': lq_cond} if self.opt['network_g']['params']['cond_lq'] else None
                    
                    # 简化：直接用MSE结果作为预测
                    x_pred = micro_sr_mse
                
                # Heteroscedastic loss (修改版3): 去除强制σ=1的正则项
                # 新设计: 让σ作为自适应权重,鼓励空间变化
                # 关键: 不再使用(σ-1)²惩罚,避免σ退化为常数
                
                # 1. Clamp防止极端值
                sigma_clamped = torch.clamp(sigma_pred, min=0.3, max=3.0)
                sigma_sq = sigma_clamped ** 2
                residual_sq = (micro_gt - x_pred) ** 2
                
                # 2. 数据拟合项: 高σ区域允许更大误差
                data_term = residual_sq / (2 * sigma_sq + 1e-6)
                
                # 3. **NEW v5**: 增强的空间多样性约束 - 渐进式目标 + 反向惩罚
                # 策略: 降低初始目标(0.15→0.05),使用双重惩罚机制
                spatial_std = torch.std(sigma_pred, dim=[2, 3], keepdim=True)  # per-image std
                
                # 惩罚1: Soft Lower Bound - 渐进式目标
                # 先让std达到0.05,训练稳定后再提高到0.15
                min_std_threshold = 0.05  # 🔧 降低初始目标 (从0.15→0.05)
                diversity_penalty_per_image = torch.relu(min_std_threshold - spatial_std) ** 2
                
                # 惩罚2: Uniformity Penalty - 反向激励
                # 思想: std越小,惩罚越大 (1/std形式)
                # 这会强制网络增加空间变化
                uniformity_penalty = 1.0 / (spatial_std + 1e-3)  # 防止除零
                
                # 4. **NEW**: 防止σ过大的上界约束
                # 使用Soft Upper Bound: 只在mean(σ)>2.0时惩罚
                mean_sigma = torch.mean(sigma_clamped)
                magnitude_penalty = torch.relu(mean_sigma - 2.0) ** 2
                
                # 5. 组合损失 (使用lambda_diversity权重)
                # 🔧 NEW v5: 添加uniformity_penalty,双管齐下
                diversity_penalty = torch.mean(diversity_penalty_per_image)
                het_loss = (torch.mean(data_term) + 
                           self.lambda_diversity * diversity_penalty + 
                           0.1 * torch.mean(uniformity_penalty) +  # 新增反向惩罚
                           0.5 * magnitude_penalty)
                
                # DEBUG: 第一次迭代时打印调试信息
                if current_iter == 1 and jj == 0:
                    logger = get_root_logger()
                    logger.info(f"[DEBUG] Heteroscedastic Loss (Modified v5 - Enhanced Diversity):")
                    logger.info(f"  sigma_pred range: [{sigma_pred.min().item():.6f}, {sigma_pred.max().item():.6f}]")
                    logger.info(f"  sigma_clamped range: [{sigma_clamped.min().item():.6f}, {sigma_clamped.max().item():.6f}]")
                    logger.info(f"  data_term: {data_term.mean().item():.6f}")
                    logger.info(f"  spatial_std: {spatial_std.mean().item():.6f} (target: ≥0.05)")
                    logger.info(f"  diversity_penalty ReLU(0.05-std)²: {diversity_penalty.item():.6f}")
                    logger.info(f"  uniformity_penalty 1/(std+1e-3): {torch.mean(uniformity_penalty).item():.6f}")
                    logger.info(f"  magnitude_penalty ReLU(mean-2.0)²: {magnitude_penalty.item():.6f}")
                    logger.info(f"  het_loss (total): {het_loss.item():.6f}")
                
                het_loss_total += het_loss
            
            # === 损失3：SR损失指导（端到端优化）===
            if self.use_sr_guidance:
                # 扩展sigma到3通道
                sigma_3ch = sigma_pred.expand_as(micro_sr_mse)
                
                # 准备条件
                lq_cond = nn.PixelUnshuffle(self.sf)(
                    torch.cat([micro_sr_mse, micro_lq_bicubic], dim=1)
                )
                model_kwargs = {'lq': lq_cond} if self.opt['network_g']['params']['cond_lq'] else None
                
                # 使用小的时间步进行前向+反向
                # 这样可以让U²M学习"有用的"不确定性
                max_t = min(self.sr_guidance_steps * 50, self.base_diffusion.num_timesteps // 4)
                tt = torch.randint(
                    0, 
                    max(1, max_t),  # 确保至少为1
                    size=(micro_gt.shape[0],),
                    device=self.lq.device
                )
                
                # 使用扩散模型的q_sample进行前向加噪
                # q_sample(x_start, y, y_hat, un, t, noise)
                noise = torch.randn_like(micro_gt)
                
                # 使用base_diffusion的q_sample（支持不确定性调制）
                x_t = self.base_diffusion.q_sample(
                    self.base_diffusion.encode_first_stage(micro_gt),
                    self.base_diffusion.encode_first_stage(micro_lq_bicubic),
                    self.base_diffusion.encode_first_stage(micro_sr_mse),
                    self.base_diffusion.encode_first_stage(sigma_3ch),  # 使用U²M预测的sigma
                    tt,
                    noise=self.base_diffusion.encode_first_stage(noise)
                )
                
                # 网络去噪预测（保持梯度）
                x_t_scaled = self.base_diffusion._scale_input(
                    x_t,
                    self.base_diffusion.encode_first_stage(sigma_3ch),
                    tt
                )
                model_output = self.net_g(x_t_scaled, tt, **model_kwargs)
                
                # 解码输出
                model_output = self.base_diffusion.decode_first_stage(model_output)
                
                # SR重建损失：预测应该接近GT
                sr_loss = F.l1_loss(model_output, micro_gt)
                sr_loss = self.lambda_sr * sr_loss
                sr_loss_total += sr_loss
                
            # 总损失
            batch_loss = reg_loss
            if self.use_heteroscedastic:
                batch_loss += het_loss
            if self.use_sr_guidance:
                batch_loss += sr_loss
            
            u2m_loss_total += batch_loss
        
        # 反向传播
        u2m_loss_total.backward()
        self.optimizer_u2m.step()
        
        # 记录损失
        loss_dict['l_u2m'] = u2m_loss_total
        loss_dict['l_u2m_reg'] = u2m_reg_total
        if self.use_heteroscedastic:
            loss_dict['l_u2m_het'] = het_loss_total
        if self.use_sr_guidance:
            loss_dict['l_u2m_sr'] = sr_loss_total
    
    def _train_uncertainty_network(self, current_iter, loss_dict):
        """训练不确定性网络"""
        self.optimizer_uncertainty.zero_grad()
        
        current_batchsize = self.lq.shape[0]
        micro_batchsize = self.opt['datasets']['train']['micro_batchsize']
        
        uncertainty_loss_total = 0.0
        
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_lq = self.lq[jj:jj+micro_batchsize,]
            micro_gt = self.gt[jj:jj+micro_batchsize,]
            
            with torch.no_grad():
                micro_lq_bicubic = torch.nn.functional.interpolate(
                    micro_lq, scale_factor=self.sf, mode='bicubic', align_corners=False
                )
                micro_sr_mse = (self.net_mse(micro_lq * 0.5 + 0.5) - 0.5) / 0.5
            
            # 前向传播获取所有输出
            outputs = self.net_uncertainty(
                micro_lq, micro_sr_mse, micro_lq_bicubic, 
                return_features=True
            )
            
            # 准备监督信号
            targets = {}
            
            # 1. 边缘图
            with torch.no_grad():
                gt_edge = compute_edge_map(micro_gt)
                gt_texture = compute_texture_complexity(micro_gt)
            targets['gt_edge'] = gt_edge
            targets['gt_texture'] = gt_texture
            
            # 2. 预测误差（通过快速前向传播）
            # 使用当前模型预测并计算误差
            with torch.no_grad():
                # 使用不确定性进行采样
                raw_uncertainty = outputs['uncertainty']
                noise_scale = self.noise_scheduler(raw_uncertainty, smooth=True)
                # 扩展到3通道以匹配图像
                noise_scale_3ch = noise_scale.expand_as(micro_sr_mse)
                
                # 简化的扩散预测（仅用于监督信号）
                tt_sample = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_gt.shape[0],),
                    device=self.lq.device
                )
                
                lq_cond = nn.PixelUnshuffle(self.sf)(
                    torch.cat([micro_sr_mse, micro_lq_bicubic], dim=1)
                )
                model_kwargs = {'lq': lq_cond} if self.opt['network_g']['params']['cond_lq'] else None
                
                # 获取x_t
                noise_sample = torch.randn_like(micro_sr_mse)
                x_t_sample = self.base_diffusion.q_sample(
                    self.base_diffusion.encode_first_stage(micro_gt),
                    self.base_diffusion.encode_first_stage(micro_lq_bicubic),
                    self.base_diffusion.encode_first_stage(micro_sr_mse),
                    self.base_diffusion.encode_first_stage(noise_scale_3ch),  # 使用3通道版本
                    tt_sample,
                    noise=self.base_diffusion.encode_first_stage(noise_sample)
                )
                
                # 预测
                x0_pred_sample = self.net_g(
                    self.base_diffusion._scale_input(
                        x_t_sample, 
                        self.base_diffusion.encode_first_stage(noise_scale_3ch),  # 使用3通道版本
                        tt_sample
                    ),
                    tt_sample,
                    **model_kwargs
                )
                
                # 计算预测误差
                pred_error = torch.abs(x0_pred_sample - micro_gt)
            
            targets['prediction_error'] = pred_error
            
            # 计算损失
            un_loss, un_loss_dict = self.uncertainty_loss(outputs, targets)
            
            uncertainty_loss_total += un_loss
        
        # 反向传播
        uncertainty_loss_total.backward()
        self.optimizer_uncertainty.step()
        
        # 记录损失
        loss_dict['l_uncertainty'] = uncertainty_loss_total
        if hasattr(self, 'uncertainty_loss'):
            for key, value in un_loss_dict.items():
                loss_dict[f'un_{key}'] = value

    def sample_func(self, y0, noise_repeat=False, return_uncertainty=False):
        desired_min_size = self.opt['val']['desired_min_size']
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        y_bicubic = torch.nn.functional.interpolate(
            y0, scale_factor=self.sf, mode='bicubic', align_corners=False,
            )
        
        y_hat = (self.net_mse(y0 * 0.5 + 0.5) - 0.5) / 0.5
        
        # === 推理时的不确定性预测 ===
        raw_uncertainty_map = None  # 用于保存单通道不确定性图
        if self.use_u2m and hasattr(self, 'net_u2m') and self.net_u2m is not None:
            # 使用U²M模块 - 正向阶段：输入 [LR_upsampled, SR_pre]
            self.net_u2m.eval()
            with torch.no_grad():
                # 将LR上采样到HR尺寸
                y0_up = torch.nn.functional.interpolate(
                    y0, size=y_hat.shape[2:],
                    mode='bilinear', align_corners=False
                )
                u2m_input = torch.cat([y0_up, y_hat], dim=1)  # (B, 6, H, W)
                sigma_fwd, global_v_inf = self.net_u2m(u2m_input)  # (B, 1, H, W), (B, 1, 1, 1)
                
                # 保存单通道不确定性图用于可视化
                if return_uncertainty:
                    raw_uncertainty_map = sigma_fwd.clone()
                
                # 扩展到3通道以匹配扩散过程
                un = sigma_fwd.expand_as(y_hat)
                
        elif self.use_uncertainty_net and hasattr(self, 'net_uncertainty') and self.net_uncertainty is not None:
            # 使用训练好的不确定性网络（旧版本）
            self.net_uncertainty.eval()
            self.noise_scheduler.eval()
            with torch.no_grad():
                raw_uncertainty = self.net_uncertainty(
                    y0, y_hat, y_bicubic, 
                    return_features=False
                )
                un = self.noise_scheduler(raw_uncertainty, smooth=True)
                # 保存单通道不确定性图用于可视化
                if return_uncertainty:
                    raw_uncertainty_map = un.clone()  # [B, 1, H, W]
                # 扩展到3通道以匹配扩散过程
                un = un.expand_as(y_hat)
        else:
            # 使用原始固定计算
            if self.opt['diffusion']['un'] > 0:
                diff = (y_hat - y_bicubic) / 2
                un_max = self.opt['diffusion']['un']
                b_un = self.opt['diffusion']['min_noise']
                un = torch.abs(diff).clamp_(0., un_max) / un_max
                un = b_un + (1 - b_un) * un
                if return_uncertainty:
                    raw_uncertainty_map = un[:, :1, :, :].clone()  # 取第一个通道
            else:
                un = torch.ones_like(y_hat)
                if return_uncertainty:
                    raw_uncertainty_map = torch.ones_like(y_hat[:, :1, :, :])

        lq_cond = nn.PixelUnshuffle(self.sf)(torch.cat([y_hat, y_bicubic], dim=1))

        model_kwargs={'lq':lq_cond,} if self.opt['network_g']['params']['cond_lq'] else None
        
        # 准备动态U2M所需的额外参数(训练和推理都启用)
        # 已修复: gaussian_diffusion中使用PixelShuffle解码到RGB空间,U2M现在可以正常工作
        # 测试时需检查net_u2m是否存在
        if (self.use_u2m and hasattr(self, 'net_u2m') and self.net_u2m is not None 
            and self.opt['diffusion'].get('use_dynamic_u2m', False)):
            model_kwargs['net_u2m'] = self.net_u2m
            model_kwargs['y0'] = y0  # 原始LR输入
            model_kwargs['sf'] = self.sf
            model_kwargs['u2m_noise_scale'] = self.opt['diffusion'].get('u2m_noise_scale', 0.1)
            model_kwargs['u2m_apply_interval'] = self.opt['diffusion'].get('u2m_apply_interval', 1)
            # 传递衰减调度器（如果有）
            if hasattr(self, 'decay_scheduler') and self.decay_scheduler is not None:
                self.decay_scheduler.eval()  # 推理时设为eval模式
                model_kwargs['decay_scheduler'] = self.decay_scheduler
            else:
                # 退回到固定衰减
                model_kwargs['u2m_noise_decay'] = self.opt['diffusion'].get('u2m_noise_decay', 'linear')
        
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g
        results = self.base_diffusion.ddim_sample_loop(
                y=y_bicubic,
                y_hat=y_hat,
                un=un,
                model=net,
                first_stage_model=None,
                noise=None,
                noise_repeat=noise_repeat,
                # clip_denoised=(self.autoencoder is None),
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                one_step=self.opt['diffusion'].get('one_step', False),
                )    

        if flag_pad:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]
            if return_uncertainty and raw_uncertainty_map is not None:
                raw_uncertainty_map = raw_uncertainty_map[:, :, :ori_h*self.sf, :ori_w*self.sf]

        results = results.clamp_(-1.0, 1.0)
        
        if return_uncertainty:
            return results, raw_uncertainty_map
        else:
            return results

    def test(self):
        # 清空CUDA缓存,防止内存碎片化导致OOM
        torch.cuda.empty_cache()

        def _process_per_image(im_lq_tensor, return_uncertainty=False):
            if im_lq_tensor.shape[2] > self.opt['val']['chop_size'] or im_lq_tensor.shape[3] > self.opt['val']['chop_size']:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.opt['val']['chop_size'],
                        stride=self.opt['val']['chop_stride'],
                        sf=self.opt['scale'],
                        extra_bs=self.opt['val']['chop_bs'],
                        )
                
                # 如果需要返回不确定性图，也需要分块处理
                if return_uncertainty:
                    uncertainty_spliter = ImageSpliterTh(
                            im_lq_tensor,
                            self.opt['val']['chop_size'],
                            stride=self.opt['val']['chop_stride'],
                            sf=self.opt['scale'],
                            extra_bs=self.opt['val']['chop_bs'],
                            )
                
                for im_lq_pch, index_infos in im_spliter:
                    if return_uncertainty:
                        im_sr_pch, uncertainty_pch = self.sample_func(
                                (im_lq_pch - 0.5) / 0.5,
                                noise_repeat=self.opt['val']['noise_repeat'],
                                return_uncertainty=True,
                                )     # 1 x c x h x w, [-1, 1]
                        im_spliter.update(im_sr_pch, index_infos)
                        uncertainty_spliter.update(uncertainty_pch, index_infos)
                    else:
                        im_sr_pch = self.sample_func(
                                (im_lq_pch - 0.5) / 0.5,
                                noise_repeat=self.opt['val']['noise_repeat'],
                                )     # 1 x c x h x w, [-1, 1]
                        im_spliter.update(im_sr_pch, index_infos)
                
                im_sr_tensor = im_spliter.gather()
                uncertainty_tensor = uncertainty_spliter.gather() if return_uncertainty else None
            else:
                if return_uncertainty:
                    im_sr_tensor, uncertainty_tensor = self.sample_func(
                            (im_lq_tensor - 0.5) / 0.5,
                            noise_repeat=self.opt['val']['noise_repeat'],
                            return_uncertainty=True,
                            )     # 1 x c x h x w, [-1, 1]
                else:
                    im_sr_tensor = self.sample_func(
                            (im_lq_tensor - 0.5) / 0.5,
                            noise_repeat=self.opt['val']['noise_repeat'],
                            )     # 1 x c x h x w, [-1, 1]
                    uncertainty_tensor = None

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            
            if return_uncertainty:
                return im_sr_tensor, uncertainty_tensor
            else:
                return im_sr_tensor
        
        # 检查是否需要保存不确定性图（支持U²M和旧的不确定性网络）
        save_uncertainty = self.opt['val'].get('save_uncertainty', False) and (self.use_u2m or self.use_uncertainty_net)
        
        if save_uncertainty:
            self.output, self.uncertainty = _process_per_image(self.lq, return_uncertainty=True)
        else:
            self.output = _process_per_image(self.lq, return_uncertainty=False)
            self.uncertainty = None  # 确保没有残留值


    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data, training=True):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if training and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # USM sharpen the GT images
            if self.opt['degradation']['use_sharp'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['degradation']['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['degradation']['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['degradation']['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['degradation']['gray_noise_prob']
            if np.random.uniform() < self.opt['degradation']['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['degradation']['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['degradation']['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['degradation']['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['degradation']['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['degradation']['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['degradation']['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['degradation']['scale'] * scale), int(ori_w / self.opt['degradation']['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['degradation']['gray_noise_prob2']
            if np.random.uniform() < self.opt['degradation']['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['degradation']['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['degradation']['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['degradation']['scale'], ori_w // self.opt['degradation']['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['degradation']['scale'], ori_w // self.opt['degradation']['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['degradation']['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['degradation']['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
            # normalize
            # if self.mean is not None or self.std is not None:
            self.lq = (self.lq - 0.5) / 0.5
            self.gt = (self.gt - 0.5) / 0.5
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
            else:
                self.gt = None

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 清空CUDA缓存,防止内存碎片化
        torch.cuda.empty_cache()
        
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                # 初始化所有实际要计算的指标
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                # 如果有PI指标,确保niqe在metric_results中(简化版PI=NIQE)
                if 'pi' in self.opt['val']['metrics']:
                    if 'niqe' not in self.metric_results:
                        self.metric_results['niqe'] = 0
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            # 重置所有指标
            for metric in self.metric_results.keys():
                self.metric_results[metric] = 0

        metric_data = dict()
        if use_pbar:
            pbar = tqdm.tqdm(total=len(dataloader), unit='image')

        num_img = 0

        for idx, val_data in enumerate(dataloader):
            num_img += len(val_data['lq_path'])
            self.feed_data(val_data, training=False)
            
            self.test()
            
            # 每个样本处理完后清理缓存
            torch.cuda.empty_cache()

            metric_data['img'] = self.output.clamp(0, 1)
            # metric_data['img'] = torch.clamp((self.output * 255.0).round(), 0, 255) / 255.
            metric_data['img2'] = self.gt

            if with_metrics:
                # calculate metrics
                if metric_data['img2'] is not None:
                    for name, metric in self.metrics_fr.items():
                        # Skip FID metric as it requires folder paths, not tensors
                        if name.lower() == 'fid':
                            continue
                        # 只计算在metric_results中的指标
                        if name in self.metric_results:
                            self.metric_results[name] += metric(metric_data['img'], metric_data['img2']).sum().item()
                for name, metric in self.metrics_nr.items():
                    # Skip FID metric as it requires folder paths, not tensors
                    if name.lower() == 'fid':
                        continue
                    # 只计算在metric_results中的指标(包括pi需要的ma)
                    if name in self.metric_results:
                        self.metric_results[name] += metric(metric_data['img']).sum().item()

            visuals = self.get_current_visuals()

            sr_img = [tensor2img(visuals['result'][ii]) for ii in range(self.output.shape[0])]
            
            # === 准备不确定性图（U²M或旧不确定性网络） ===
            uncertainty_imgs = None
            if hasattr(self, 'uncertainty') and self.uncertainty is not None:
                # 将不确定性图转换为可视化热力图
                uncertainty_imgs = []
                
                # 🔧 使用固定范围归一化 (避免自适应归一化的误导)
                # 对于U²M: 期望σ在 [sigma_min, sigma_max] 范围
                # 对于旧网络: 使用 [min_noise, max_noise] 范围
                if self.use_u2m:
                    # 从配置读取sigma范围(推理时使用train配置或默认值)
                    train_opt = self.opt.get('train', {})
                    sigma_min = train_opt.get('u2m_loss_opt', {}).get('sigma_min', 0.01)
                    sigma_max = train_opt.get('u2m_loss_opt', {}).get('sigma_max', 0.45)
                    # 扩展上限以显示过大的σ
                    vis_min, vis_max = sigma_min, sigma_max * 2.0
                else:
                    # 旧网络使用扩散配置的范围
                    vis_min = self.opt['diffusion'].get('min_noise', 0.4)
                    vis_max = 1.0
                
                for ii in range(self.uncertainty.shape[0]):
                    # uncertainty: [B, 1, H, W]
                    un_map = self.uncertainty[ii, 0].detach().cpu().numpy()  # [H, W]
                    
                    # 固定范围归一化 (不再自适应)
                    # σ < vis_min → 蓝色 (0.0)
                    # σ ≈ (vis_min + vis_max)/2 → 绿色 (0.5)
                    # σ > vis_max → 红色 (1.0)
                    un_map_norm = np.clip((un_map - vis_min) / (vis_max - vis_min + 1e-8), 0.0, 1.0)
                    
                    # 转换为热力图
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.cm as cm
                    
                    # 使用 jet 色图（蓝色=低不确定性，红色=高不确定性）
                    colored = cm.jet(un_map_norm)[:, :, :3]  # 去掉alpha通道
                    colored = (colored * 255).astype('uint8')
                    uncertainty_imgs.append(colored)
                    
                    # 📊 添加统计信息到日志
                    if idx == 0 and ii == 0:  # 只记录第一个batch的第一张图
                        logger = get_root_logger()
                        logger.info(f"[Validation] Uncertainty map stats: "
                                  f"min={un_map.min():.4f}, max={un_map.max():.4f}, "
                                  f"mean={un_map.mean():.4f}, std={un_map.std():.4f} "
                                  f"(vis_range=[{vis_min:.2f}, {vis_max:.2f}])")
            
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                # metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if hasattr(self, 'uncertainty') and self.uncertainty is not None:
                del self.uncertainty

            torch.cuda.empty_cache()

            
            for ii in range(len(val_data['lq_path'])):
                if save_img:
                    img_name = osp.splitext(osp.basename(val_data['lq_path'][ii]))[0]

                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.png')
                        # 保存不确定性图
                        if uncertainty_imgs is not None:
                            save_uncertainty_path = osp.join(self.opt['path']['visualization'], img_name,
                                                            f'{img_name}_{current_iter}_uncertainty.png')
                            imwrite(uncertainty_imgs[ii], save_uncertainty_path)
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                            # 保存不确定性图
                            if uncertainty_imgs is not None:
                                save_uncertainty_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                                f'{img_name}_{self.opt["val"]["suffix"]}_uncertainty.png')
                                imwrite(uncertainty_imgs[ii], save_uncertainty_path)
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["name"]}.png')
                            # 保存不确定性图
                            if uncertainty_imgs is not None:
                                save_uncertainty_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                                f'{img_name}_{self.opt["name"]}_uncertainty.png')
                                imwrite(uncertainty_imgs[ii], save_uncertainty_path)
                            
                    imwrite(sr_img[ii], save_img_path)
                    
            
            if use_pbar:
                pbar.update(1)

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= num_img
                # update the best metric result (只更新配置中明确要求的指标)
                if metric in self.opt['val']['metrics']:
                    self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            
            # Calculate PI if requested (simplified: PI = NIQE)
            # 注: 原始PI = 0.5*((10-MA)+NIQE), 由于MA兼容性问题,简化为PI=NIQE
            if 'pi' in self.opt['val']['metrics']:
                if 'niqe' in self.metric_results:
                    pi_val = self.metric_results['niqe']  # 简化版本: 直接使用NIQE
                    self.metric_results['pi'] = pi_val
                    # update the best metric result for PI
                    self._update_best_metric_result(dataset_name, 'pi', pi_val, current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
            # === Early Stopping 检查 ===
            if self.early_stopping_enabled and dataset_name == 'Val_SR':
                self._check_early_stopping(current_iter)
    
    def _check_early_stopping(self, current_iter):
        """检查是否触发Early Stopping"""
        logger = get_root_logger()
        
        # 获取监控指标的当前值
        if self.early_stopping_metric not in self.metric_results:
            logger.warning(f"Early stopping metric '{self.early_stopping_metric}' not found in results")
            return
        
        current_value = self.metric_results[self.early_stopping_metric]
        
        # 初始化最佳值
        if self.early_stopping_best_value is None:
            self.early_stopping_best_value = current_value
            self.early_stopping_best_iter = current_iter
            logger.info(f"[Early Stopping] Initial {self.early_stopping_metric}: {current_value:.4f}")
            return
        
        # 检查是否有改进
        improved = False
        if self.early_stopping_mode == 'max':
            if current_value > self.early_stopping_best_value + self.early_stopping_min_delta:
                improved = True
        else:  # min
            if current_value < self.early_stopping_best_value - self.early_stopping_min_delta:
                improved = True
        
        if improved:
            # 有改进，重置计数器
            improvement = current_value - self.early_stopping_best_value
            self.early_stopping_best_value = current_value
            self.early_stopping_best_iter = current_iter
            self.early_stopping_counter = 0
            logger.info(f"[Early Stopping] ✅ {self.early_stopping_metric} improved: "
                       f"{self.early_stopping_best_value-improvement:.4f} → {current_value:.4f} "
                       f"(+{improvement:.4f})")
        else:
            # 无改进，增加计数器
            self.early_stopping_counter += 1
            logger.info(f"[Early Stopping] ⚠️ No improvement in {self.early_stopping_metric}: "
                       f"current={current_value:.4f}, best={self.early_stopping_best_value:.4f} @ iter {self.early_stopping_best_iter}")
            logger.info(f"[Early Stopping] Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
            
            # 检查是否达到patience
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.warning("=" * 80)
                logger.warning(f"🛑 EARLY STOPPING TRIGGERED!")
                logger.warning(f"Best {self.early_stopping_metric}: {self.early_stopping_best_value:.4f} @ iter {self.early_stopping_best_iter}")
                logger.warning(f"No improvement for {self.early_stopping_patience} consecutive validations ({self.early_stopping_patience * 2000} iters)")
                logger.warning(f"Training will stop. Please use checkpoint from iter {self.early_stopping_best_iter}")
                logger.warning("=" * 80)
                
                # 触发停止标志
                import sys
                sys.exit(0)  # 优雅退出训练

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt') and self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        """保存网络权重，包括U²M和不确定性网络"""
        # 保存主网络
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        
        # 保存U²M网络
        if self.use_u2m and hasattr(self, 'net_u2m') and self.net_u2m is not None:
            logger = get_root_logger()
            logger.info(f"Saving U2M network at iter {current_iter}")
            self.save_network(self.net_u2m, 'net_u2m', current_iter, param_key='u2m')
        
        # 保存不确定性网络（旧版本，保持兼容）
        elif self.use_uncertainty_net and hasattr(self, 'net_uncertainty') and self.net_uncertainty is not None:
            logger = get_root_logger()
            logger.info(f"Saving uncertainty network at iter {current_iter}")
            # 保存不确定性网络和噪声调度器
            self.save_network(
                [self.net_uncertainty, self.noise_scheduler], 
                'net_uncertainty', 
                current_iter, 
                param_key=['uncertainty_net', 'noise_scheduler']
            )
        
        # 保存训练状态
        self.save_training_state(epoch, current_iter)
