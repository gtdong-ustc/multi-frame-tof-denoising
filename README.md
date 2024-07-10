# ECCV2024_Exploiting Dual-Correlation for Multi-frame Time-of-Flight Denoising

This repository provides the source code of multi-frame time-of-flight (ToF) noise removal.

## How to use the code

code running environment
```
tensorflow-gpu==1.12.0 
```

the project starts with the 'start.py'. Through this file, you can select different models, data sets and loss functions for training, and you can switch between train, eval and output modes by adjusting parameters
The parameters available are as follows

```
Arg
├───modelName		# Select the model required during training
│   ├───sample_pyramid_add_kpn                 # SHARP-Net
│   ├───dear_kpn_no_rgb                        # ToF-KPN
│   └───dear_kpn_no_rgb_DeepToF                # DeepToF
├───trainingSet		# Select the dataset required during training
│   ├───tof_FT3       # ToF-FlyingThings3D dataset (480, 640)
│   ├───HAMMER          # HAMMER dataset (424, 512)
│   └───CornellBox       # CornellBox dataset (640, 640)
├───flagMode		# Select the running mode of the code
│   ├───train                 # train model
│   ├───eval_ED               # evaluate model in test sets
│   ├───eval_TD               # evaluate model in training sets
│   └───output                # output depth prediction, offsets, weight
├───gpuNumber		# The number of GPU used in training
├───addGradient		# weather add the gradient loss function
├───decayEpoch		# after n epochs, decay the learning rate
├───lossType		# Select the loss function in training
│   ├───mean_l1               # the mean of L1 loss between input and gt
│   ├───mean_l2               # the mean of L2 loss
│   ├───sum_l1                # the sum of L1 loss
│   ├───sum_l2                # the sum of L2 loss
│   ├───smoothness            # depth map smoothness loss
│   ├───SSIM                  # the sum of structural similarity loss 
│   ├───SSIM_l1               # the sum of structural similarity loss + L1 loss
│   ├───mean_SSIM             # the mean of structural similarity loss 
│   ├───ZNCC                  # Zero Mean Normalized Cross-Correlation loss
│   ├───cos_similarity        # cos_similarity
│   ├───mean_huber            # the mean of huber loss
│   └───sum_huber             # the sum of huber loss
└───lossMask	    # Select the loss mask to be used during training
    ├───gt_msk                # Non-zero region in groundtruth
    ├───depth_kinect_msk      # Non-zero region in depth input
    └───depth_kinect_with_gt_msk      # gt_msk with depth_kinect_msk
```
For example

```
python start.py -b 2 -s 200000 -m pyramid_corr_multi_frame_denoising -p size384 -k depth_kinect_with_gt_msk -l 0.0004 -t tof_FT3 -i 480 640 -o mean_l1 --addGradient sobel_gradient -g 4 -e 1200
```


   

