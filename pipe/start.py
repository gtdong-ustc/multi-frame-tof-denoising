# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# this code simulates the time-of-flight data
# all time unit are picoseconds (1 picosec = 1e-12 sec)
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
sys.path.insert(0, '../sim/')
import argparse
import tensorflow as tf
from PIL import Image


tf.logging.set_verbosity(tf.logging.INFO)
from loss import *
from model import *
from kinect_pipeline import *
from dataset import *
from metric import *

# tof_cam = kinect_real_tf()

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def tof_net_func(features, labels, mode, params):
    """
    This is the network function of tensorflow estimator API
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """
    depth_kinect = None
    depth_kinect_msk = None
    amplitude_kinect = None
    rgb_kinect = None
    raw_new = None
    depth_residual_every_scale = None
    gt_msk = None
    loss_mask_dict = {}

    if params['training_set'] == 'tof_FT3':
        if params['output_flg'] == True:
            gt = None
            gt_msk = features['noisy'] > 10.0/4095.0
            gt_msk = tf.cast(gt_msk, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        else:
            gt = labels['gt']
            gt_msk = gt > 1e-4
            gt_msk = tf.cast(gt_msk, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        full = features['noisy']
        intensity = features['intensity']
        rgb = features['rgb']

        depth_kinect = full
        amplitude_kinect = intensity
        rgb_kinect = rgb
        depth_kinect_msk = depth_kinect < 1.0
        depth_kinect_msk_tmp = depth_kinect > 10.0/4095.0
        depth_kinect_msk = tf.cast(depth_kinect_msk, tf.float32)
        depth_kinect_msk_tmp = tf.cast(depth_kinect_msk_tmp, tf.float32)
        depth_kinect_msk = depth_kinect_msk * depth_kinect_msk_tmp

    elif params['training_set'] == 'TB':
        if params['output_flg'] == True:
            gt = None
            gt_msk = None
            loss_mask_dict['gt_msk'] = gt_msk
        else:
            gt = labels['gt']
            gt_msk = gt > 1e-4
            gt_msk = tf.cast(gt_msk, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        full = features['noisy']
        intensity = features['intensity']
        rgb = features['rgb']

        depth_kinect = full
        amplitude_kinect = intensity
        rgb_kinect = rgb
        depth_kinect_msk = depth_kinect < 2.0
        depth_kinect_msk_tmp = depth_kinect > 1e-4
        depth_kinect_msk = tf.cast(depth_kinect_msk, tf.float32)
        depth_kinect_msk_tmp = tf.cast(depth_kinect_msk_tmp, tf.float32)
        depth_kinect_msk = depth_kinect_msk * depth_kinect_msk_tmp

    elif params['training_set'] == 'RGBDD':
        if params['output_flg'] == True:
            gt = None
        else:
            gt = labels['gt']
            gt_msk = gt > 1e-4
            gt_msk = tf.cast(gt_msk, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        full = features['noisy']
        intensity = features['intensity']
        rgb = features['rgb']

        depth_kinect = full
        amplitude_kinect = intensity
        rgb_kinect = rgb
        depth_kinect_msk = depth_kinect < 1.0
        depth_kinect_msk_tmp = depth_kinect > 10.0/4095.0
        depth_kinect_msk = tf.cast(depth_kinect_msk, tf.float32)
        depth_kinect_msk_tmp = tf.cast(depth_kinect_msk_tmp, tf.float32)
        depth_kinect_msk = depth_kinect_msk * depth_kinect_msk_tmp

    elif params['training_set'] == 'cornellbox':
        if params['output_flg'] == True:
            gt = None
        else:
            gt = labels['gt']
            gt_msk = gt_msk = tf.cast(gt < 1e3, tf.float32) * tf.cast(gt > 0.0000001, tf.float32)
            # gt_msk = tf.cast(gt_msk, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        full = features['noisy']
        amplitude = features['amplitude']

        depth_kinect = full
        amplitude_kinect = amplitude
        depth_kinect_msk = gt_msk
        # depth_kinect_msk = tf.cast(depth_kinect_msk, tf.float32)

    elif params['training_set'] == 'FLAT':
        if params['output_flg'] == True:
            gt = None
        else:
            gt = labels['gt']
            # gt_msk = (gt < 1e3) * (gt != 0)
            gt_msk = tf.cast(gt < 1e3, tf.float32) * tf.cast(gt > 0.00001, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        full = features['noisy']
        amplitude = features['amplitude']

        depth_kinect = full
        amplitude_kinect = amplitude
        depth_kinect_msk = gt_msk
        # depth_kinect_msk = tf.cast(depth_kinect_msk, tf.float32)

    elif params['training_set'] == 'Agresti_S1':
        if params['output_flg'] == True:
            gt = None
        else:
            gt = labels['gt']
            # gt_msk = tf.cast(gt != 0, tf.float32)
            gt_msk = tf.cast(gt < 1e3, tf.float32) * tf.cast(gt > 0.0000001, tf.float32)
            # gt_msk = tf.cast(gt_msk, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        full = features['noisy']
        amplitude = features['amplitude']
        depth_kinect = full
        amplitude_kinect = amplitude
        depth_kinect_msk = gt_msk
        # depth_kinect_msk = depth_kinect > 1e-4
        # depth_kinect_msk = tf.cast(depth_kinect_msk, tf.float32)

    ### add kinect_msk
    loss_mask_dict['depth_kinect_msk'] = depth_kinect_msk
    if gt_msk != None:
        # loss_mask_dict['gt_msk'] = gt_msk * depth_kinect_msk
        loss_mask_dict['depth_kinect_with_gt_msk'] = gt_msk * depth_kinect_msk
        loss_mask_dict['gt_msk'] = gt_msk
    else:
        loss_mask_dict['depth_kinect_with_gt_msk'] = depth_kinect_msk

    model_name_list = params['model_name'].split('_')

    if params['training_set'] == 'tof_FT3':
        inputs = tf.concat([depth_kinect, amplitude_kinect, rgb_kinect], axis=-1)
    elif params['training_set'] == 'TB':
        inputs = tf.concat([depth_kinect, amplitude_kinect, rgb_kinect], axis=-1)
    elif params['training_set'] == 'RGBDD':
        inputs = tf.concat([depth_kinect, amplitude_kinect, rgb_kinect], axis=-1)
    elif params['training_set'] == 'FLAT':
        inputs = tf.stack([depth_kinect[:, :, :, 1],
                           depth_kinect[:,:,:,2] - depth_kinect[:,:,:,1],
                           depth_kinect[:,:,:,0] - depth_kinect[:,:,:,1],
                           amplitude_kinect[:,:,:,2] / amplitude_kinect[:,:,:,1] - 1,
                           amplitude_kinect[:,:,:,0] / amplitude_kinect[:,:,:,1] - 1], -1)
    elif params['training_set'] == 'cornellbox':
        inputs = tf.stack([depth_kinect[:, :, :, 2],
                           depth_kinect[:, :, :, 0] - depth_kinect[:, :, :, 2],
                           depth_kinect[:, :, :, 1] - depth_kinect[:, :, :, 2],
                           amplitude_kinect[:, :, :, 0] / amplitude_kinect[:, :, :, 2] - 1,
                           amplitude_kinect[:, :, :, 1] / amplitude_kinect[:, :, :, 2] - 1], -1)
    elif params['training_set'] == 'Agresti_S1':
        inputs = tf.stack([depth_kinect[:, :, :, 2],
                           depth_kinect[:, :, :, 0] - depth_kinect[:, :, :, 2],
                           depth_kinect[:, :, :, 1] - depth_kinect[:, :, :, 2],
                           amplitude_kinect[:, :, :, 0] / amplitude_kinect[:, :, :, 2] - 1,
                           amplitude_kinect[:, :, :, 1] / amplitude_kinect[:, :, :, 2] - 1], -1)
    else:
        inputs = depth_kinect

    depth_outs, depth_residual_every_scale = get_network(name=params['model_name'], x=inputs, flg=mode == tf.estimator.ModeKeys.TRAIN,
                             regular=0.1, batch_size=params['batch_size'], range=params['deformable_range'])

    ## get the msk needed in compute loss and metrics
    loss_msk = loss_mask_dict[params['loss_mask']]

    # compute loss (for TRAIN and EVAL modes)
    loss = None
    train_op = None
    metrics = None
    evaluation_hooks = []

    if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):

        if params['add_gradient'] == 'sobel_gradient':
            loss_iters_w = [0.7, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
            # loss_iters_w = [0.75, 0.75, 0.75, 0.75]
            loss_1 = get_supervised_loss(params['loss_fn'], depth_outs, gt, loss_msk)
            loss_2 = get_supervised_loss('sobel_gradient', depth_outs * loss_msk, gt * loss_msk, loss_msk)
            loss = loss_1 + 10.0 * loss_2
            # for i in range(len(depth_residual_every_scale)):
            #     w = loss_iters_w[i]
            #     loss_1 = get_supervised_loss(params['loss_fn'], depth_residual_every_scale[i], gt, loss_msk)
            #     loss_2 = get_supervised_loss('sobel_gradient', depth_residual_every_scale[i], gt, loss_msk)
            #     loss += w * (loss_1 + 10.0 * loss_2)
                # loss += w * loss_1
        else:
            loss = get_supervised_loss(params['loss_fn'], depth_outs, gt, loss_msk)
        # configure the training op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('training_loss', loss)
            global_step = tf.train.get_global_step()
            learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step=global_step, decay_steps= params['decay_epoch'] * int(params['samples_number'] / params['batch_size']), decay_rate=0.8)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)
        if mode == tf.estimator.ModeKeys.EVAL:
            if loss_msk == None:
                amplitude_kinect_map = tf.identity(amplitude_kinect, 'amplitude_kinect')
                depth_gt_map = tf.identity(gt, 'depth_gt')
                depth_outs_map = tf.identity(depth_outs, 'depth_outs')
                depth_kinect_map = tf.identity(depth_kinect, 'depth_kinect')
                depth_outs_error = tf.identity(gt - depth_outs, 'depth_outs_error')
                depth_kinect_error = tf.identity(gt - depth_kinect, 'depth_kinect_error')
                depth_kinect_error_negative = depth_kinect_error * tf.cast(depth_kinect_error < 0, dtype=tf.float32)
                tensor_min = tf.reduce_min(depth_kinect_error_negative, axis=[1, 2, 3], keepdims=True)
                depth_kinect_error_negative = depth_kinect_error_negative / tensor_min
                depth_kinect_error_positive = depth_kinect_error * tf.cast(depth_kinect_error >= 0, dtype=tf.float32)
                tensor_max = tf.reduce_max(depth_kinect_error_negative, axis=[1, 2, 3], keepdims=True)
                depth_kinect_error_positive = depth_kinect_error_positive / tensor_max
                depth_outs_error_negative = depth_outs_error * tf.cast(depth_outs_error < 0, dtype=tf.float32)
                depth_outs_error_negative = depth_outs_error_negative / tensor_min
                depth_outs_error_positive = depth_outs_error * tf.cast(depth_outs_error >= 0, dtype=tf.float32)
                depth_outs_error_positive = depth_outs_error_positive / tensor_max
            else:
                depth_gt_map = tf.identity(gt, 'depth_gt')
                amplitude_kinect_map = tf.identity(amplitude_kinect, 'amplitude_kinect')
                depth_outs_map = tf.identity(depth_outs * loss_msk, 'depth_outs')
                depth_kinect_map = tf.identity(depth_kinect, 'depth_kinect')
                depth_outs_error = tf.identity((gt * loss_msk - depth_outs * loss_msk), 'depth_outs_error')
                depth_kinect_error = tf.identity((gt * loss_msk - depth_kinect * loss_msk), 'depth_kinect_error')
                depth_kinect_error_negative = depth_kinect_error * tf.cast(depth_kinect_error < 0, dtype=tf.float32)
                tensor_min = tf.reduce_min(depth_kinect_error_negative, axis=[1, 2, 3], keepdims=True)
                depth_kinect_error_negative = depth_kinect_error_negative / tensor_min
                depth_kinect_error_positive = depth_kinect_error * tf.cast(depth_kinect_error >= 0, dtype=tf.float32)
                tensor_max = tf.reduce_max(depth_kinect_error_negative, axis=[1, 2, 3], keepdims=True)
                depth_kinect_error_positive = depth_kinect_error_positive / tensor_max
                depth_outs_error_negative = depth_outs_error * tf.cast(depth_outs_error < 0, dtype=tf.float32)
                depth_outs_error_negative = depth_outs_error_negative / tensor_min
                depth_outs_error_positive = depth_outs_error * tf.cast(depth_outs_error >= 0, dtype=tf.float32)
                depth_outs_error_positive = depth_outs_error_positive / tensor_max


            tf.summary.image('depth_gt', colorize_img(depth_gt_map, vmin=0.0, vmax=1.0, cmap='jet'))
            # tf.summary.image('amplitude_kinect_map', amplitude_kinect_map)
            # tf.summary.image('depth_outs', colorize_img(depth_outs_map, vmin=0.0, vmax=1.0, cmap='jet'))
            # tf.summary.image('depth_kinect', colorize_img(depth_kinect_map, vmin=0.0, vmax=1.0, cmap='jet'))
            # tf.summary.image('depth_outs_error', colorize_img(tf.abs(depth_outs_error), vmin=0.0, vmax=0.1, cmap='jet'))
            # tf.summary.image('depth_kinect_error',colorize_img(tf.abs(depth_kinect_error), vmin=0.0, vmax=0.4, cmap='jet'))
            # tf.summary.image('depth_outs_error_positive', colorize_img(depth_outs_error_positive, vmin=0.0, vmax=1.0, cmap='jet'))
            # tf.summary.image('depth_kinect_error_positive', colorize_img(depth_kinect_error_positive, vmin=0.0, vmax=1.0, cmap='jet'))
            # tf.summary.image('depth_outs_error_negative',colorize_img(depth_outs_error_negative, vmin=0.0, vmax=1.0, cmap='jet'))
            # tf.summary.image('depth_kinect_error_negative',colorize_img(depth_kinect_error_negative, vmin=0.0, vmax=1.0, cmap='jet'))
            ## get metrics
            if params['training_set'] == 'tof_FT3':
                depth_outs = depth_outs * 409.5
                depth_kinect = depth_kinect * 409.5
                gt = gt * 409.5
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs, depth_kinect , gt, loss_msk)
            elif params['training_set'] == 'RGBDD':
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs, depth_kinect , gt, loss_msk)
            elif params['training_set'] == 'cornellbox':
                depth_kinect_test2 = tf.expand_dims(depth_kinect[:, :, :, 1], -1) * 100.0
                depth_kinect_test1 = tf.expand_dims(depth_kinect[:, :, :, 2], -1)
                depth_outs = depth_outs * 100.0
                depth_kinect_test1 = depth_kinect_test1 * 100.0
                gt_test = gt * 100.0
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(
                    depth_outs, depth_kinect_test1, gt_test, loss_msk)
            elif params['training_set'] == 'FLAT':
                depth_kinect = tf.expand_dims(depth_kinect[:,:,:,1], -1)
                depth_outs = depth_outs * 100.0
                depth_kinect = depth_kinect * 100.0
                gt = gt * 100.0
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(
                    depth_outs, depth_kinect, gt, loss_msk)
            elif params['training_set'] == 'Agresti_S1':
                # depth_kinect = tf.expand_dims(depth_kinect[:,:,:,1],-1) + tf.expand_dims(depth_kinect[:,:,:,0],-1)
                depth_kinect = tf.expand_dims(depth_kinect[:, :, :, 2], -1)
                depth_outs = depth_outs * 100.0
                depth_kinect = depth_kinect * 100.0
                # msk = tf.cast(gt > 0.0000001, tf.float32)
                gt = gt * 100.0
                # gt_test = tf.zeros_like(gt)
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(
                    depth_outs, depth_kinect, gt, loss_msk)
            elif params['training_set'] == 'TB':
                depth_outs = depth_outs * 200.0
                depth_kinect = depth_kinect * 200.0
                gt = gt * 200.0
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(
                    depth_outs, depth_kinect, gt, loss_msk)

            metrics = {
                "ori_MAE": ori_mae,
                "pre_MAE": pre_mae,
                'pre_mae_percent_25':pre_mae_percent_25,
                'pre_mae_percent_50':pre_mae_percent_50,
                'pre_mae_percent_75':pre_mae_percent_75,
            }
            eval_summary_hook = tf.train.SummarySaverHook(
                                save_steps=1,
                                output_dir= params['model_dir'] + "/eval",
                                summary_op=tf.summary.merge_all())
            evaluation_hooks.append(eval_summary_hook)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics,
                evaluation_hooks=evaluation_hooks
            )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "depth_input": depth_kinect,
            "irs_input": amplitude_kinect,
            "depth": depth_outs,
            # "offset": offsets
        }
    else:
        predictions = None

    # return a ModelFnOps object
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
    )

def dataset_training(train_data_path, evaluate_data_path, model_dir, loss_fn, learning_rate, batch_size, traing_steps,
                     evaluate_steps, deformable_range, model_name, checkpoint_steps, loss_mask, gpu_Number,
                     training_set, image_shape, samples_number, add_gradient, decay_epoch):
    """
    This function represents the training mode of the code
    :param train_data_path:
    :param evaluate_data_path:
    :param model_dir:
    :param loss_fn:
    :param learning_rate:
    :param batch_size:
    :param traing_steps:
    :param evaluate_steps:
    :param deformable_range:
    :param model_name:
    :param checkpoint_steps:
    :param loss_mask:
    :param gpu_Number:
    :param training_set:
    :param image_shape:
    :return: no return
    """
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=gpu_Number)
    # session_config = tf.ConfigProto(device_count=gpu_device_number_list[gpu_Number - 1])
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        keep_checkpoint_max=10,
        save_checkpoints_steps=evaluate_steps,
        # session_config=session_config,
        save_summary_steps=50,
        log_step_count_steps=20,
        train_distribute=strategy,
    )  # set the frequency of logging steps for loss function

    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration,
                                     params={'learning_rate': learning_rate, 'batch_size': batch_size, 'model_dir': model_dir,
                                             'loss_fn': loss_fn, 'deformable_range': deformable_range, 'model_name': model_name,
                                             'loss_mask': loss_mask, 'output_flg':False, 'training_set': training_set,
                                             'samples_number': samples_number, 'add_gradient': add_gradient, 'decay_epoch':decay_epoch})

    print('##############################')
    # print(sum([np.prod(tof_net.get_variable_value(var).shape) for var in tof_net.get_variable_names()]))

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: get_input_fn(training_set=training_set, filenames=train_data_path, height=image_shape[0], width=image_shape[1],
                                                               shuffle=True, repeat_count=-1, batch_size=batch_size),
                                        max_steps=traing_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                                                             shuffle=False, repeat_count=1, batch_size=batch_size),
                                      steps=None, throttle_secs=evaluate_steps)
    tf.estimator.train_and_evaluate(tof_net, train_spec, eval_spec)
def dataset_testing(evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range,
                    loss_fn, model_name, loss_mask, gpu_Number, training_set, image_shape, add_gradient):
    """
    This function represents the eval mode of the code
    :param evaluate_data_path:
    :param model_dir:
    :param batch_size:
    :param checkpoint_steps:
    :param deformable_range:
    :param loss_fn:
    :param model_name:
    :param loss_mask:
    :param gpu_Number:
    :param training_set:
    :param image_shape:
    :return:
    """
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=gpu_Number)
    # session_config = tf.ConfigProto(device_count=gpu_device_number_list[gpu_Number - 1])
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        # session_config=session_config,
        log_step_count_steps = 10,
        save_summary_steps = 5,
        train_distribute=strategy,
    )
    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration,
        params={'learning_rate': 1e-4, 'batch_size': batch_size, 'model_dir': model_dir,
                'deformable_range': deformable_range, 'loss_fn':loss_fn,'add_gradient':add_gradient,
                'model_name': model_name, 'loss_mask': loss_mask, 'output_flg':False, 'training_set': training_set})

    tof_net.evaluate(input_fn=lambda: get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
        shuffle=False, repeat_count=1, batch_size=batch_size), checkpoint_path=model_dir + '/model.ckpt-' + checkpoint_steps)

def dataset_output(result_path, evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range,
                   loss_fn, model_name, loss_mask, gpu_Number, training_set, image_shape):
    """
    This function represents the output mode of the code
    :param result_path:
    :param evaluate_data_path:
    :param model_dir:
    :param batch_size:
    :param checkpoint_steps:
    :param deformable_range:
    :param loss_fn:
    :param model_name:
    :param loss_mask:
    :param gpu_Number:
    :param training_set:
    :param image_shape:
    :return:
    """
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=gpu_Number)
    # session_config = tf.ConfigProto(device_count=gpu_device_number_list[gpu_Number - 1])
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        # session_config=session_config,
        log_step_count_steps = 10,
        save_summary_steps = 5,
        train_distribute=strategy,
    )
    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration,
        params={'learning_rate': 1e-4, 'batch_size': batch_size, 'model_dir': model_dir,
                'deformable_range': deformable_range, 'loss_fn':loss_fn,
                'model_name': model_name, 'loss_mask': loss_mask, 'output_flg':True, 'training_set': training_set})
    result = list(tof_net.predict(input_fn=lambda: get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                shuffle=False, repeat_count=1, batch_size=batch_size), checkpoint_path=model_dir + '/model.ckpt-' + checkpoint_steps))

    root_dir = result_path
    pre_depth_dir = os.path.join(root_dir, 'pre_depth')
    depth_input_dir = os.path.join(root_dir, 'depth_input')
    amplitude_dir = os.path.join(root_dir, 'amplitude')
    depth_input_png_dir = os.path.join(root_dir, 'depth_input_png')
    pre_depth_png_dir = os.path.join(root_dir, 'pre_depth_png')
    amplitude_png_dir = os.path.join(root_dir, 'amplitude_png')
    if not os.path.exists(pre_depth_dir):
        os.mkdir(pre_depth_dir)
    if not os.path.exists(depth_input_dir):
        os.mkdir(depth_input_dir)
    if not os.path.exists(amplitude_dir):
        os.mkdir(amplitude_dir)
    if not os.path.exists(depth_input_png_dir):
        os.mkdir(depth_input_png_dir)
    if not os.path.exists(pre_depth_png_dir):
        os.mkdir(pre_depth_png_dir)
    if not os.path.exists(amplitude_png_dir):
        os.mkdir(amplitude_png_dir)

    for i in range(len(result)):
        pre_depth_path = os.path.join(pre_depth_dir, str(i))
        depth_input_path = os.path.join(depth_input_dir, str(i))
        amplitude_path = os.path.join(amplitude_dir, str(i))
        depth_input_png_path = os.path.join(depth_input_png_dir, str(i)+'.png')
        pre_depth_png_path = os.path.join(pre_depth_png_dir, str(i) + '.png')
        amplitude_png_path = os.path.join(amplitude_png_dir, str(i) + '.png')
        pre_depth = np.squeeze(result[i]['depth'])
        input_depth = np.squeeze(result[i]['depth_input'])
        amplitude = np.squeeze(result[i]['irs_input'])
        input_depth_png = input_depth * 100
        pre_depth_png = pre_depth * 100
        amplitude_png = amplitude * 255

        print(input_depth_png.shape)
        pre_depth = np.reshape(pre_depth, -1).astype(np.float32)
        input_depth = np.reshape(input_depth, -1).astype(np.float32)
        amplitude = np.reshape(amplitude, -1).astype(np.float32)

        input_depth_png = Image.fromarray(input_depth_png)
        input_depth_png = input_depth_png.convert("L")
        input_depth_png.save(depth_input_png_path)

        pre_depth_png = Image.fromarray(pre_depth_png)
        pre_depth_png = pre_depth_png.convert("L")
        pre_depth_png.save(pre_depth_png_path)

        amplitude_png = Image.fromarray(amplitude_png)
        amplitude_png = amplitude_png.convert("L")
        amplitude_png.save(amplitude_png_path)

        amplitude.tofile(amplitude_path)
        pre_depth.tofile(pre_depth_path)
        input_depth.tofile(depth_input_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training of a Deformable KPN Network')
    parser.add_argument("-t", "--trainingSet", help='the name to the list file with training set', default = 'FLAT_reflection_s5', type=str)
    parser.add_argument("-m", "--modelName", help="name of the denoise model to be used", default="deformable_kpn")
    parser.add_argument("-l", "--lr", help="initial value for learning rate", default=1e-5, type=float)
    parser.add_argument("-i", "--imageShape", help='two int for image shape [height,width]', nargs='+', type=int, default=[239, 320])##default=[424, 512]
    parser.add_argument("-b", "--batchSize", help='batch size to use during training', type=int, default=4)
    parser.add_argument("-s", "--steps", help='number of training steps', type=int, default=4000)
    parser.add_argument("-e", "--evalSteps", help='after the number of training steps to eval', type=int, default=400)
    parser.add_argument("-o", '--lossType', help="Type of supervised loss to use, such as mean_l2, mean_l1, sum_l2, sum_l1, smoothness, SSIM", default="mean_l2", type=str)
    parser.add_argument("-d", "--deformableRange", help="the range of deformable kernel", default=192.0, type=float)
    parser.add_argument("-f", '--flagMode', help="The flag that select the runing mode, such as train, eval, output", default='train', type=str)
    parser.add_argument("-p", '--postfix', help="the postfix of the training task", default=None, type=str)
    parser.add_argument("-c", '--checkpointSteps', help="select the checkpoint of the model", default="800", type=str)
    parser.add_argument("-k", '--lossMask', help="the mask used in compute loss", default='gt_msk', type=str)
    parser.add_argument("-g", '--gpuNumber', help="The number of GPU used in training", default=2, type=int)
    parser.add_argument('--samplesNumber', help="samples number in one epoch", default=5800, type=int)
    parser.add_argument('--addGradient', help="add the gradient loss function", default='sobel_gradient', type=str)
    parser.add_argument('--decayEpoch', help="after n epoch, decay the learning rate", default=2, type=int)
    parser.add_argument('--shmFlag', help="using shm increase the training speed", default=False, type=bool)
    args = parser.parse_args()

    if args.shmFlag == True:
        dataset_dir = '/dev/shm/dataset/tfrecords'
    else:
        # dataset_dir = '/data/1015323606/RGB-D-D/tfrecords'
        dataset_dir = '/data/1015323606/ToFFlyingThings3D/tfrecords'
    model_dir = '/data/1015323606/ToFFlyingThings3D/ToFDenoisingTF_pretrain/' + args.modelName
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if args.modelName[0:10] == 'deformable':
        mkdir_name = args.trainingSet + '_' + args.lossType + '_dR' + str(args.deformableRange)
    # elif args.trainingSet == 'TB':
    #     mkdir_name = 'tof_FT3' + '_' + args.lossType
    else:
        mkdir_name = args.trainingSet + '_' + args.lossType
    if args.postfix is not None:
        model_dir = os.path.join(model_dir, mkdir_name + '_' + args.postfix)
    else:
        model_dir = os.path.join(model_dir, mkdir_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    output_dir = os.path.join(model_dir, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dataset_path = os.path.join(dataset_dir, args.trainingSet)
    train_data_path = os.path.join(dataset_path, args.trainingSet + '_train.tfrecords')
    evaluate_data_path = os.path.join(dataset_path, args.trainingSet + '_eval.tfrecords')


    if args.flagMode == 'train':
        dataset_training(train_data_path=train_data_path, evaluate_data_path=evaluate_data_path, loss_fn=args.lossType,
                         model_dir=model_dir, learning_rate=args.lr, batch_size=args.batchSize, traing_steps=args.steps,
                         evaluate_steps=args.evalSteps, deformable_range = args.deformableRange, model_name=args.modelName,
                         checkpoint_steps=args.checkpointSteps, loss_mask = args.lossMask, gpu_Number = args.gpuNumber,
                         training_set = args.trainingSet, image_shape = args.imageShape, samples_number = args.samplesNumber,
                         add_gradient = args.addGradient, decay_epoch=args.decayEpoch)
    elif args.flagMode == 'eval_TD':
        dataset_testing(evaluate_data_path=train_data_path, model_dir=model_dir, loss_fn=args.lossType,batch_size=args.batchSize,
                        checkpoint_steps=args.checkpointSteps, deformable_range = args.deformableRange, model_name = args.modelName,
                        loss_mask = args.lossMask, gpu_Number = args.gpuNumber, training_set = args.trainingSet, image_shape = args.imageShape,
                        add_gradient=args.addGradient)
    elif args.flagMode == 'eval_ED':
        dataset_testing(evaluate_data_path=evaluate_data_path, model_dir=model_dir, loss_fn=args.lossType,
                        # batch_size=args.batchSize,
                        batch_size=args.batchSize,
                        checkpoint_steps=args.checkpointSteps, deformable_range=args.deformableRange,
                        model_name=args.modelName,
                        loss_mask=args.lossMask, gpu_Number=args.gpuNumber, training_set=args.trainingSet,
                        image_shape=args.imageShape,
                        add_gradient=args.addGradient)
    else:
        dataset_output(result_path=output_dir,evaluate_data_path=evaluate_data_path, model_dir=model_dir, loss_fn=args.lossType,
                        batch_size=args.batchSize, checkpoint_steps=args.checkpointSteps, deformable_range = args.deformableRange,
                        model_name = args.modelName, loss_mask = args.lossMask, gpu_Number = args.gpuNumber, training_set = args.trainingSet,
                        image_shape = args.imageShape)