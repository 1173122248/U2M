# 计算DISTS、PI、FID指标使用示例

# 示例1: 计算所有三个指标
python meters/calculate_dists_pi_fid.py ^
  --sr_path results/000_UPSR_RealSR_x4_archived_20251126_105211/visualization/LSDIR-test ^
  --gt_path D:\DataSets\LSDIR-test\HR\val ^
  --metrics dists pi fid ^
  --save_json meters/results_all.json

# 示例2: 只计算DISTS
python meters/calculate_dists_pi_fid.py ^
  --sr_path results/000_UPSR_RealSR_x4_archived_20251126_105211/visualization/LSDIR-test ^
  --gt_path D:\DataSets\LSDIR-test\HR\val ^
  --metrics dists ^
  --save_json meters/results_dists.json

# 示例3: 只计算PI (包含NIQE和MA)
python meters/calculate_dists_pi_fid.py ^
  --sr_path results/000_UPSR_RealSR_x4_archived_20251126_105211/visualization/LSDIR-test ^
  --gt_path D:\DataSets\LSDIR-test\HR\val ^
  --metrics pi ^
  --save_json meters/results_pi.json

# 示例4: 只计算FID
python meters/calculate_dists_pi_fid.py ^
  --sr_path results/000_UPSR_RealSR_x4_archived_20251126_105211/visualization/LSDIR-test ^
  --gt_path D:\DataSets\LSDIR-test\HR\val ^
  --metrics fid ^
  --save_json meters/results_fid.json

# 示例5: 指定文件后缀 (如果SR文件名有特殊后缀)
python meters/calculate_dists_pi_fid.py ^
  --sr_path results/xxx/visualization/dataset ^
  --gt_path D:\DataSets\xxx\HR ^
  --metrics dists pi fid ^
  --suffix _sr ^
  --save_json meters/results.json

# 示例6: 使用JPG格式
python meters/calculate_dists_pi_fid.py ^
  --sr_path results/xxx/visualization/dataset ^
  --gt_path D:\DataSets\xxx\HR ^
  --metrics dists pi fid ^
  --pattern "*.jpg" ^
  --save_json meters/results.json
