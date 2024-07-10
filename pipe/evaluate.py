import numpy as np
import os

root_dir = '/data/1015323606/ToFFlyingThings3D/ToFDenoisingTF_pretrain/sample_pyramid_add_kpn_recurrent_v5_7/tof_FT3_mean_l1_size384_1080Ti_T3/output/'
depth_input_dir = os.path.join(root_dir, 'depth_input')
depth_gt_dir = os.path.join(root_dir, 'depth_gt')
depth_pre_dir = os.path.join(root_dir, 'depth_pre')
irs_input_dir = os.path.join(root_dir, 'irs_input')
results_txt_path = os.path.join(root_dir, 'eval.txt')
if os.path.exists(results_txt_path):
    os.remove(results_txt_path)
all_mae = 0.0
all_pre_mae = 0.0
for num in range(119):
    num_str = str(num)
    depth_input_path = os.path.join(depth_input_dir, num_str)
    depth_gt_path = os.path.join(depth_gt_dir, num_str)
    depth_pre_path = os.path.join(depth_pre_dir, num_str)
    irs_input_path = os.path.join(irs_input_dir, num_str)

    depth_input = np.fromfile(depth_input_path, dtype=np.float32)
    depth_gt = np.fromfile(depth_gt_path, dtype=np.float32)
    depth_pre = np.fromfile(depth_pre_path, dtype=np.float32)
    irs_input = np.fromfile(irs_input_path, dtype=np.float32)

    depth_input = np.reshape(depth_input, (384, 512)).astype(np.float32)
    depth_gt = np.reshape(depth_gt, (384, 512)).astype(np.float32)
    depth_pre = np.reshape(depth_pre, (384, 512)).astype(np.float32)
    irs_input = np.reshape(irs_input, (384, 512)).astype(np.float32)

    input_mae = np.abs(depth_input * 4.095 - depth_gt * 4.095).mean()
    pre_mae = np.abs(depth_pre * 4.095 - depth_gt * 4.095).mean()
    # print(depth_input)

    write_str = 'Scene '+num_str+', Original MAE: '+ str(input_mae)+', MAE: '+ str(pre_mae) + '\n'

    with open(results_txt_path, 'a') as f:
        f.write(write_str)

    print(write_str)
    all_mae += input_mae
    all_pre_mae += pre_mae
    # print(depth_input.shape)
all_write_str = 'Scene All'+', Original MAE: '+ str(all_mae/119.0)+', MAE: '+ str(all_pre_mae/119.0) + '\n'
with open(results_txt_path, 'a') as f:
    f.write(all_write_str)
print(all_write_str)


