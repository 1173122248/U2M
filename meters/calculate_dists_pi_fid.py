"""
计算DISTS、PI、FID指标
独立脚本，不修改原有代码

使用示例:
# 有GT图的情况(可计算全参考指标DISTS/FID和非参考指标PI)
python meters/calculate_dists_pi_fid.py --sr_path results/test_UPSR_U2M_x4/visualization/RealSRV3 --gt_path F:/ZZM/UPSR-main/basicsr/datasets/RealSRV3/HR --metrics dists pi fid --suffix _test_UPSR_U2M_x4

# 无GT图的情况(只计算非参考指标PI/NIQE/MA)
python meters/calculate_dists_pi_fid.py --sr_path results/test_UPSR_U2M_x4/visualization/RealSet65 --metrics pi --suffix _test_UPSR_U2M_x4

参数说明:
--sr_path: 超分结果目录(必需)
--gt_path: Ground Truth目录(可选,无GT时只能计算PI等非参考指标)
--metrics: 要计算的指标 (dists, pi, fid) - 无GT时只能选pi
--save_json: 保存结果到JSON文件
"""

import argparse
import os
import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import cv2
import torch
import pyiqa
import sys

# 添加父目录到path以导入basicsr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.utils import img2tensor


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate DISTS, PI, and FID metrics')
    
    parser.add_argument('--sr_path', type=str, required=True,
                        help='Path to SR results directory')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to ground truth directory (optional, if not provided, only no-reference metrics like PI can be calculated)')
    
    parser.add_argument('--metrics', nargs='+', 
                        default=['dists', 'pi', 'fid'],
                        choices=['dists', 'pi', 'fid'],
                        help='Metrics to calculate')
    
    parser.add_argument('--pattern', type=str, default='*.png',
                        help='File pattern (e.g., *.png, *.jpg)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix to remove from SR filename')
    
    parser.add_argument('--save_json', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-image results')
    
    return parser.parse_args()


def load_image(img_path):
    """加载图像为RGB格式"""
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    
    # BGR to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def find_matching_files(sr_path, gt_path=None, pattern='*.png', suffix=''):
    """查找SR和GT匹配的文件对(gt_path可选,无GT时只返回SR文件)"""
    sr_path = Path(sr_path)
    sr_files = sorted(sr_path.glob(pattern))
    
    if not sr_files:
        raise ValueError(f"No SR files found in {sr_path} with pattern {pattern}")
    
    file_pairs = []
    
    # 如果没有GT路径,只返回SR文件
    if gt_path is None:
        for sr_file in sr_files:
            file_pairs.append({
                'sr': str(sr_file),
                'name': sr_file.stem,
                'gt': None
            })
        return file_pairs
    
    gt_path_obj = Path(gt_path)
    
    if not gt_path_obj.is_absolute():
        gt_path_obj = gt_path_obj.resolve()
    
    for sr_file in sr_files:
        name = sr_file.stem
        original_name = name
        
        # 去除后缀
        if suffix and name.endswith(suffix):
            name = name[:-len(suffix)]
        
        # 自动检测SR后缀 (如 0000001_000_UPSR_RealSR_x4 -> 0000001)
        if not suffix and '_' in name:
            base_name = name.split('_')[0]
        else:
            base_name = name
        
        pair = {
            'sr': str(sr_file),
            'name': original_name,
            'gt': None
        }
        
        # 尝试匹配GT
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIF']
        name_candidates = [name, base_name]
        
        for candidate_name in name_candidates:
            for ext in extensions:
                gt_file = gt_path_obj / f"{candidate_name}{ext}"
                if gt_file.exists():
                    pair['gt'] = str(gt_file)
                    pair['name'] = candidate_name
                    break
            if pair['gt'] is not None:
                break
        
        if pair['gt'] is None:
            print(f"Warning: No GT found for {sr_file.name}")
        
        file_pairs.append(pair)
    
    return file_pairs


def calculate_dists(sr_images, gt_images, device='cuda'):
    """
    计算DISTS (Deep Image Structure and Texture Similarity) 指标
    使用pyiqa库实现，与原UPSR项目一致
    
    DISTS是全参考指标，越低表示质量越好
    范围: [0, ∞)，通常在[0, 1]区间
    
    Args:
        sr_images: SR图像列表 (numpy arrays, RGB, uint8)
        gt_images: GT图像列表 (numpy arrays, RGB, uint8)
        device: 计算设备 ('cuda' or 'cpu')
    
    Returns:
        list: 每张图像的DISTS分数
    """
    print("\n[DISTS] Calculating with pyiqa...")
    
    # 使用pyiqa创建DISTS指标（与原UPSR项目一致）
    dists_metric = pyiqa.create_metric('dists', device=device)
    
    scores = []
    for sr_img, gt_img in tqdm(zip(sr_images, gt_images), total=len(sr_images), desc="DISTS"):
        # 转换为tensor，范围[0,1]（pyiqa标准输入格式）
        sr_tensor = img2tensor(sr_img, bgr2rgb=False, float32=True).unsqueeze(0).to(device) / 255.0
        gt_tensor = img2tensor(gt_img, bgr2rgb=False, float32=True).unsqueeze(0).to(device) / 255.0
        
        # 计算DISTS（全参考，需要SR和GT）
        score = dists_metric(sr_tensor, gt_tensor).item()
        scores.append(score)
        
        torch.cuda.empty_cache()
    
    return scores


def calculate_pi(sr_images, device='cuda'):
    """
    计算PI (Perceptual Index) 指标
    标准PI公式: PI = 0.5 × ((10 - MA) + NIQE)
    
    注意: pyiqa不直接支持MA指标，这里使用musiq-ava作为近似
    musiq-ava输出范围[0,1]，需要×10转换为[0,10]
    
    Returns:
        dict: {'pi': list, 'niqe': list, 'ma': list}
    """
    print("\n[PI] Calculating NIQE and MA (musiq-ava)...")
    
    # 使用pyiqa创建指标（与原UPSR项目一致）
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    ma_metric = pyiqa.create_metric('musiq-ava', device=device)
    
    niqe_scores = []
    ma_scores = []
    pi_scores = []
    
    for sr_img in tqdm(sr_images, desc="PI (NIQE+MA)"):
        # 转换为tensor，范围[0,1]（pyiqa标准输入）
        sr_tensor = img2tensor(sr_img, bgr2rgb=False, float32=True).unsqueeze(0).to(device) / 255.0
        
        # 计算NIQE(越低越好,范围通常0-100+)
        niqe = niqe_metric(sr_tensor).item()
        
        # 计算MA (musiq-ava直接输出0-10范围的美学分数)
        ma = ma_metric(sr_tensor).item()
        
        # 计算PI: PI = 0.5 × ((10 - MA) + NIQE)
        # 越低越好,表示感知质量越高
        pi = 0.5 * ((10.0 - ma) + niqe)
        
        niqe_scores.append(niqe)
        ma_scores.append(ma)
        pi_scores.append(pi)
        
        torch.cuda.empty_cache()
    
    return {
        'pi': pi_scores,
        'niqe': niqe_scores,
        'ma': ma_scores
    }


def calculate_fid(sr_images, gt_images, device='cuda', batch_size=32):
    """
    计算FID (Fréchet Inception Distance)
    
    注意: pyiqa不支持FID指标，这里使用标准的Inception V3实现
    FID测量SR和GT在Inception特征空间的分布距离
    
    FID = ||μ_sr - μ_gt||² + Tr(Σ_sr + Σ_gt - 2√(Σ_sr·Σ_gt))
    
    Args:
        sr_images: SR图像列表
        gt_images: GT图像列表
        device: 计算设备
        batch_size: 批处理大小
    
    Returns:
        float: FID分数（越低越好，0表示完全相同）
    """
    print("\n[FID] Calculating with Inception V3 (this may take a while)...")
    print("Note: pyiqa does not support FID, using standard torchvision implementation")
    
    try:
        from scipy import linalg
        import torch.nn.functional as F
        from torchvision.models import inception_v3
        
        # 加载预训练的Inception V3模型
        print("Loading Inception V3...")
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        inception.fc = torch.nn.Identity()  # 移除最后的全连接层，只提取特征
        
        def extract_features(images, model, batch_size=32):
            """
            提取Inception V3特征
            
            Inception V3要求:
            - 输入尺寸: 299×299
            - 输入范围: [-1, 1]
            - 输出: 2048维特征向量
            """
            features = []
            
            for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
                batch_imgs = images[i:i+batch_size]
                batch_tensors = []
                
                for img in batch_imgs:
                    # 转换为tensor [0,1]
                    tensor = img2tensor(img, bgr2rgb=False, float32=True).to(device) / 255.0
                    # Resize到299×299 (Inception V3标准输入大小)
                    tensor = F.interpolate(tensor.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False)
                    # 归一化到[-1, 1] (Inception V3标准范围)
                    tensor = tensor * 2.0 - 1.0
                    batch_tensors.append(tensor)
                
                batch_tensor = torch.cat(batch_tensors, dim=0)
                
                with torch.no_grad():
                    feat = model(batch_tensor)  # [batch_size, 2048]
                    features.append(feat.cpu().numpy())
                
                torch.cuda.empty_cache()
            
            return np.concatenate(features, axis=0)  # [N, 2048]
        
        # 提取特征
        print("Extracting SR features...")
        sr_features = extract_features(sr_images, inception, batch_size)
        
        print("Extracting GT features...")
        gt_features = extract_features(gt_images, inception, batch_size)
        
        # 计算均值和协方差
        mu_sr = np.mean(sr_features, axis=0)
        mu_gt = np.mean(gt_features, axis=0)
        
        sigma_sr = np.cov(sr_features, rowvar=False)
        sigma_gt = np.cov(gt_features, rowvar=False)
        
        # 计算FID
        diff = mu_sr - mu_gt
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma_sr.dot(sigma_gt), disp=False)
        
        # 数值稳定性
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_sr.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma_sr + offset).dot(sigma_gt + offset))
        
        # 如果是复数，取实部
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_sr + sigma_gt - 2 * covmean)
        
        return float(fid)
        
    except Exception as e:
        print(f"FID calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    args = parse_args()
    
    # 检查是否有GT
    has_gt = args.gt_path is not None
    
    # 验证指标选择
    if not has_gt:
        invalid_metrics = [m for m in args.metrics if m in ['dists', 'fid']]
        if invalid_metrics:
            print(f"Error: Metrics {invalid_metrics} require GT images, but --gt_path not provided.")
            print("Without GT, only 'pi' (non-reference metric) can be calculated.")
            return
    
    print("="*80)
    print("Calculate DISTS, PI, and FID Metrics")
    print("="*80)
    print(f"SR Path: {args.sr_path}")
    print(f"GT Path: {args.gt_path if has_gt else 'None (no-reference mode)'}")
    print(f"Metrics: {args.metrics}")
    print(f"Mode: {'Full-reference + No-reference' if has_gt else 'No-reference only'}")
    print("="*80)
    
    # 查找文件
    print("\n[Step 1] Finding files...")
    file_pairs = find_matching_files(args.sr_path, args.gt_path, args.pattern, args.suffix)
    
    if has_gt:
        # 过滤掉没有GT的图像
        file_pairs = [p for p in file_pairs if p['gt'] is not None]
        print(f"Found {len(file_pairs)} matched SR-GT pairs")
    else:
        print(f"Found {len(file_pairs)} SR images (no-reference mode)")
    
    if len(file_pairs) == 0:
        print("Error: No files found!")
        return
    
    # 加载图像
    print("\n[Step 2] Loading images...")
    sr_images = []
    gt_images = []
    
    for pair in tqdm(file_pairs, desc="Loading"):
        try:
            sr_img = load_image(pair['sr'])
            sr_images.append(sr_img)
            
            if has_gt:
                gt_img = load_image(pair['gt'])
                
                if sr_img.shape != gt_img.shape:
                    print(f"\nWarning: Size mismatch for {pair['name']}: SR {sr_img.shape} vs GT {gt_img.shape}")
                    continue
                    
                gt_images.append(gt_img)
        except Exception as e:
            print(f"\nError loading {pair['name']}: {e}")
    
    if has_gt:
        print(f"Successfully loaded {len(sr_images)} image pairs")
    else:
        print(f"Successfully loaded {len(sr_images)} SR images")
    
    # 检测GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # 计算指标
    results = OrderedDict()
    
    # DISTS
    if 'dists' in args.metrics:
        dists_scores = calculate_dists(sr_images, gt_images, device)
        results['dists'] = {
            'per_image': dists_scores,
            'mean': float(np.mean(dists_scores)),
            'std': float(np.std(dists_scores))
        }
        print(f"\nDISTS: {results['dists']['mean']:.4f} ± {results['dists']['std']:.4f}")
    
    # PI
    if 'pi' in args.metrics:
        pi_results = calculate_pi(sr_images, device)
        results['pi'] = {
            'per_image': pi_results['pi'],
            'mean': float(np.mean(pi_results['pi'])),
            'std': float(np.std(pi_results['pi']))
        }
        results['niqe'] = {
            'per_image': pi_results['niqe'],
            'mean': float(np.mean(pi_results['niqe'])),
            'std': float(np.std(pi_results['niqe']))
        }
        results['ma'] = {
            'per_image': pi_results['ma'],
            'mean': float(np.mean(pi_results['ma'])),
            'std': float(np.std(pi_results['ma']))
        }
        print(f"\nPI: {results['pi']['mean']:.4f} ± {results['pi']['std']:.4f}")
        print(f"  NIQE: {results['niqe']['mean']:.4f} ± {results['niqe']['std']:.4f}")
        print(f"  MA: {results['ma']['mean']:.4f} ± {results['ma']['std']:.4f}")
    
    # FID
    if 'fid' in args.metrics:
        fid_score = calculate_fid(sr_images, gt_images, device)
        if fid_score is not None:
            results['fid'] = float(fid_score)
            print(f"\nFID: {results['fid']:.4f}")
    
    # 保存结果
    if args.save_json:
        output_data = {
            'metrics': results,
            'config': {
                'sr_path': args.sr_path,
                'gt_path': args.gt_path,
                'num_images': len(sr_images),
                'metrics_calculated': args.metrics
            }
        }
        
        os.makedirs(os.path.dirname(args.save_json) if os.path.dirname(args.save_json) else '.', exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[Saved] Results saved to {args.save_json}")
    
    # 打印汇总
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    if 'dists' in results:
        print(f"DISTS ↓: {results['dists']['mean']:.4f}")
    if 'pi' in results:
        print(f"PI    ↓: {results['pi']['mean']:.4f}")
        print(f"NIQE  ↓: {results['niqe']['mean']:.4f}")
        print(f"MA    ↑: {results['ma']['mean']:.4f}")
    if 'fid' in results:
        print(f"FID   ↓: {results['fid']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
