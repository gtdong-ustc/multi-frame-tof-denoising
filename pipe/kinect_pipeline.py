import sys

sys.path.insert(0, '../sim/')
import tensorflow as tf
from tof_class import *
from kinect_spec import *

tf.logging.set_verbosity(tf.logging.INFO)
from kinect_init import *

# tof_cam = kinect_real_tf()

PI = 3.14159265358979323846
flg = False
dtype = tf.float32

def kinect_pipeline(meas):
    ## Kinect Pipeline, use the algorithm of Kinect v2 to compute the denoised depth

    # convert to the default data type
    x_kinect = tf.cast(meas, tf.float32)
    # make the size to be 424,512 (padding 0)
    y_idx = int((424 - int(x_kinect.shape[1])) / 2)
    zero_mat = tf.zeros([tf.shape(x_kinect)[0], y_idx, tf.shape(x_kinect)[2], 9])
    x_kinect = tf.concat([zero_mat, x_kinect, zero_mat], 1)

    msk = kinect_mask_tensor()
    msk = tf.expand_dims(tf.expand_dims(msk, 0), -1)
    # x_kinect = x_kinect * msk * tof_cam.cam['map_max']
    x_kinect = x_kinect * msk

    # final depth prediction: kinect pipeline
    ira, irb, iramp = processPixelStage1_mat(x_kinect)
    depth_outs, ir_sum_outs, ir_outs, msk_out1 = processPixelStage2(ira, irb, iramp)

    # creates the mask
    ms = tf.concat([ira, irb, iramp], -1)
    bilateral_max_edge_tests = filterPixelStage1(ms)[1]
    depth_out_edges = depth_outs * bilateral_max_edge_tests
    msk_out2 = filterPixelStage2(depth_outs, depth_out_edges, ir_outs)[1]
    msk_out3 = tf.cast(tf.greater(depth_outs, prms['min_depth']), dtype=dtype)
    msk_out4 = tf.cast(tf.less(depth_outs, prms['max_depth']), dtype=dtype)
    depth_msk = tf.cast(tf.greater(msk_out2 * msk_out3 * msk_out4, 0.5), dtype=dtype)

    depth_outs /= 1000.0
    depth_outs *= depth_msk

    # baseline correction
    depth_outs = depth_outs * base_cor['k'] + base_cor['b']

    depth_outs = depth_outs[:, 20:-20, :]
    depth_msk = depth_msk[:, 20:-20, :]
    amplitude_outs = ir_outs[:, 20:-20, :]

    return depth_outs, depth_msk, amplitude_outs
def kinect_mask_tensor():
    # return the kinect mask that creates the positive-negative interval
    mask = np.zeros((424, 512))
    idx = 1
    for i in range(mask.shape[0]):
        mask[i, :] = idx
        if i != (mask.shape[0] / 2 - 1):
            idx = -idx

    mask = tf.convert_to_tensor(mask)
    mask = tf.cast(mask, tf.float32)
    return mask
def processPixelStage1(m):
    # m is (None,424, 512, 9)
    # the first three is the first frequency
    tmp = []
    tmp.append(processMeasurementTriple(m[:, :, :, 0:3], prms['ab_multiplier_per_frq'][0], trig_table0))
    tmp.append(processMeasurementTriple(m[:, :, :, 3:6], prms['ab_multiplier_per_frq'][1], trig_table1))
    tmp.append(processMeasurementTriple(m[:, :, :, 6:9], prms['ab_multiplier_per_frq'][2], trig_table2))

    m_out = [ \
        tmp[0][:, :, :, 0], tmp[1][:, :, :, 0], tmp[2][:, :, :, 0],
        tmp[0][:, :, :, 1], tmp[1][:, :, :, 1], tmp[2][:, :, :, 1],
        tmp[0][:, :, :, 2], tmp[1][:, :, :, 2], tmp[2][:, :, :, 2],
    ]
    m_out = tf.stack(m_out, -1)

    # return processMeasurementTriple(m[:,:,:,0:3], prms['ab_multiplier_per_frq'][0], trig_table0)
    return m_out
def processPixelStage1_mat(m):
    # if not saturated
    cos_tmp0 = np.stack([trig_table0[:, :, 0], trig_table1[:, :, 0], trig_table2[:, :, 0]], -1)
    cos_tmp1 = np.stack([trig_table0[:, :, 1], trig_table1[:, :, 1], trig_table2[:, :, 1]], -1)
    cos_tmp2 = np.stack([trig_table0[:, :, 2], trig_table1[:, :, 2], trig_table2[:, :, 2]], -1)

    sin_negtmp0 = np.stack([trig_table0[:, :, 3], trig_table1[:, :, 3], trig_table2[:, :, 3]], -1)
    sin_negtmp1 = np.stack([trig_table0[:, :, 4], trig_table1[:, :, 4], trig_table2[:, :, 4]], -1)
    sin_negtmp2 = np.stack([trig_table0[:, :, 5], trig_table1[:, :, 5], trig_table2[:, :, 5]], -1)

    # stack
    cos_tmp0 = np.expand_dims(cos_tmp0, 0)
    cos_tmp1 = np.expand_dims(cos_tmp1, 0)
    cos_tmp2 = np.expand_dims(cos_tmp2, 0)
    sin_negtmp0 = np.expand_dims(sin_negtmp0, 0)
    sin_negtmp1 = np.expand_dims(sin_negtmp1, 0)
    sin_negtmp2 = np.expand_dims(sin_negtmp2, 0)

    #
    abMultiplierPerFrq = np.expand_dims(np.expand_dims(np.expand_dims(prms['ab_multiplier_per_frq'], 0), 0), 0)

    ir_image_a = cos_tmp0 * m[:, :, :, 0::3] + cos_tmp1 * m[:, :, :, 1::3] + cos_tmp2 * m[:, :, :, 2::3]
    ir_image_b = sin_negtmp0 * m[:, :, :, 0::3] + sin_negtmp1 * m[:, :, :, 1::3] + sin_negtmp2 * m[:, :, :, 2::3]

    ir_image_a *= abMultiplierPerFrq
    ir_image_b *= abMultiplierPerFrq
    ir_amplitude = tf.sqrt(ir_image_a ** 2 + ir_image_b ** 2) * prms['ab_multiplier']

    return ir_image_a, ir_image_b, ir_amplitude
def processMeasurementTriple(m, abMultiplierPerFrq, trig_table):
    # m is (None,424,512,3)

    zmultiplier = tf.constant(z_table, dtype=dtype)

    # judge where saturation happens
    saturated = tf.cast(tf.less(tf.abs(m), 1.0), dtype=dtype)
    saturated = 1 - saturated[:, :, :, 0] * saturated[:, :, :, 1] * saturated[:, :, :, 2]

    # if not saturated
    cos_tmp0 = trig_table[:, :, 0]
    cos_tmp1 = trig_table[:, :, 1]
    cos_tmp2 = trig_table[:, :, 2]

    sin_negtmp0 = trig_table[:, :, 3]
    sin_negtmp1 = trig_table[:, :, 4]
    sin_negtmp2 = trig_table[:, :, 5]

    # stack
    cos_tmp0 = np.expand_dims(cos_tmp0, 0)
    cos_tmp1 = np.expand_dims(cos_tmp1, 0)
    cos_tmp2 = np.expand_dims(cos_tmp2, 0)
    sin_negtmp0 = np.expand_dims(sin_negtmp0, 0)
    sin_negtmp1 = np.expand_dims(sin_negtmp1, 0)
    sin_negtmp2 = np.expand_dims(sin_negtmp2, 0)

    ir_image_a = cos_tmp0 * m[:, :, :, 0] + cos_tmp1 * m[:, :, :, 1] + cos_tmp2 * m[:, :, :, 2]
    ir_image_b = sin_negtmp0 * m[:, :, :, 0] + sin_negtmp1 * m[:, :, :, 1] + sin_negtmp2 * m[:, :, :, 2]

    ir_image_a *= abMultiplierPerFrq
    ir_image_b *= abMultiplierPerFrq

    ir_amplitude = tf.sqrt(ir_image_a ** 2 + ir_image_b ** 2) * prms['ab_multiplier']

    m_out = tf.stack([ir_image_a, ir_image_b, ir_amplitude], -1)

    return m_out
def processPixelStage2(ira, irb, iramp):
    ratio = 100
    tmp0 = tf.atan2(ratio * (irb + 1e-10), ratio * (ira + 1e-10))
    flg = tf.cast(tf.less(tmp0, 0.0), dtype)
    tmp0 = flg * (tmp0 + PI * 2) + (1 - flg) * tmp0

    tmp1 = tf.sqrt(ira ** 2 + irb ** 2) * prms['ab_multiplier']

    ir_sum = tf.reduce_sum(tmp1, -1)

    # disable disambiguation
    ir_min = tf.reduce_min(tmp1, -1)

    # phase mask
    phase_msk1 = tf.cast( \
        tf.greater(ir_min, prms['individual_ab_threshold']),
        dtype=dtype
    )
    phase_msk2 = tf.cast( \
        tf.greater(ir_sum, prms['ab_threshold']),
        dtype=dtype
    )
    phase_msk_t = phase_msk1 * phase_msk2

    # compute phase
    t0 = tmp0[:, :, :, 0] / (2.0 * PI) * 3.0
    t1 = tmp0[:, :, :, 1] / (2.0 * PI) * 15.0
    t2 = tmp0[:, :, :, 2] / (2.0 * PI) * 2.0

    t5 = tf.floor((t1 - t0) * 0.3333333 + 0.5) * 3.0 + t0
    t3 = t5 - t2
    t4 = t3 * 2.0

    c1 = tf.cast(tf.greater(t4, -t4), dtype=dtype)
    f1 = c1 * 2.0 + (1 - c1) * (-2.0)
    f2 = c1 * 0.5 + (1 - c1) * (-0.5)
    t3 = t3 * f2
    t3 = (t3 - tf.floor(t3)) * f1

    c2 = tf.cast(tf.less(0.5, tf.abs(t3)), dtype=dtype) * \
         tf.cast(tf.less(tf.abs(t3), 1.5), dtype=dtype)
    t6 = c2 * (t5 + 15.0) + (1 - c2) * t5
    t7 = c2 * (t1 + 15.0) + (1 - c2) * t1
    t8 = (tf.floor((t6 - t2) * 0.5 + 0.5) * 2.0 + t2) * 0.5

    t6 /= 3.0
    t7 /= 15.0

    # transformed phase measurements (they are transformed and divided
    # by the values the original values were multiplied with)
    t9 = t8 + t6 + t7
    t10 = t9 / 3.0  # some avg

    t6 = t6 * 2.0 * PI
    t7 = t7 * 2.0 * PI
    t8 = t8 * 2.0 * PI

    t8_new = t7 * 0.826977 - t8 * 0.110264
    t6_new = t8 * 0.551318 - t6 * 0.826977
    t7_new = t6 * 0.110264 - t7 * 0.551318

    t8 = t8_new
    t6 = t6_new
    t7 = t7_new

    norm = t8 ** 2 + t6 ** 2 + t7 ** 2
    mask = tf.cast(tf.greater(t9, 0.0), dtype)
    t10 = t10

    slope_positive = float(0 < prms['ab_confidence_slope'])

    ir_min_ = tf.reduce_min(tmp1, -1)
    ir_max_ = tf.reduce_max(tmp1, -1)

    ir_x = slope_positive * ir_min_ + (1 - slope_positive) * ir_max_

    ir_x = tf.log(ir_x)
    ir_x = (ir_x * prms['ab_confidence_slope'] * 0.301030 + prms['ab_confidence_offset']) * 3.321928
    ir_x = tf.exp(ir_x)
    ir_x = tf.maximum(prms['min_dealias_confidence'], ir_x)
    ir_x = tf.minimum(prms['max_dealias_confidence'], ir_x)
    ir_x = ir_x ** 2

    mask2 = tf.cast(tf.greater(ir_x, norm), dtype)

    t11 = t10

    mask3 = tf.cast( \
        tf.greater(prms['max_dealias_confidence'] ** 2, norm),
        dtype
    )
    t10 = t10
    phase = t11

    # mask out dim regions
    phase = phase

    # phase to depth mapping
    zmultiplier = z_table
    xmultiplier = x_table

    phase_msk = tf.cast(tf.less(0.0, phase), dtype)
    phase = phase_msk * (phase + prms['phase_offset']) + (1 - phase_msk) * phase

    depth_linear = zmultiplier * phase
    depth = depth_linear
    max_depth = phase * prms['unambiguous_dist'] * 2

    cond1 = tf.cast(tf.less(0.0, depth_linear), dtype) * \
            tf.cast(tf.less(0.0, max_depth), dtype)
    depth_out = depth
    ir_sum_out = ir_sum
    ir_out = tf.minimum( \
        tf.reduce_sum(iramp, -1) * 0.33333333 * prms['ab_output_multiplier'],
        65535.0
    )

    msk_out = cond1 * phase_msk_t * mask * mask2 * mask3
    return depth_out, ir_sum_out, ir_out, msk_out
def filterPixelStage1(m):
    # m is (None, 424, 512, 9)
    # the first three is measurement a
    # the second three is measurement b
    # the third three is amplitude

    norm2 = m[:, :, :, 0:3] ** 2 + m[:, :, :, 3:6] ** 2
    inv_norm = 1.0 / tf.sqrt(norm2)

    # get rid of those nan
    inv_norm = tf.minimum(inv_norm, 1e10)

    m_normalized = tf.stack([m[:, :, :, 0:3] * inv_norm, m[:, :, :, 3:6] * inv_norm], -1)

    threshold = prms['joint_bilateral_ab_threshold'] ** 2 / prms['ab_multiplier'] ** 2
    joint_bilateral_exp = prms['joint_bilateral_exp']
    threshold = tf.constant(threshold, dtype=dtype)
    joint_bilateral_exp = tf.constant(joint_bilateral_exp, dtype=dtype)

    # set the parts with norm2 < threshold to be zero
    norm_flag = tf.cast(tf.less(norm2, threshold), dtype=dtype)
    threshold = (1 - norm_flag) * threshold
    joint_bilateral_exp = (1 - norm_flag) * joint_bilateral_exp

    # guided bilateral filtering
    gauss = prms['gaussian_kernel']
    weight_acc = tf.ones(tf.shape(m_normalized)[0:4]) * gauss[1, 1]
    weighted_m_acc0 = gauss[1, 1] * m[:, :, :, 0:3]
    weighted_m_acc1 = gauss[1, 1] * m[:, :, :, 3:6]

    # coefficient for bilateral space
    m_n = m_normalized

    # proxy for other m normalized
    m_l = tf.concat([m_n[:, :, 1::, :], m_n[:, :, 0:1, :]], 2)
    m_r = tf.concat([m_n[:, :, -1::, :], m_n[:, :, 0:-1, :]], 2)
    m_u = tf.concat([m_n[:, 1::, :, :], m_n[:, 0:1, :, :]], 1)
    m_d = tf.concat([m_n[:, -1::, :, :], m_n[:, 0:-1, :, :]], 1)
    m_lu = tf.concat([m_l[:, 1::, :, :], m_l[:, 0:1, :, :]], 1)
    m_ru = tf.concat([m_r[:, 1::, :, :], m_r[:, 0:1, :, :]], 1)
    m_ld = tf.concat([m_l[:, -1::, :, :], m_l[:, 0:-1, :, :]], 1)
    m_rd = tf.concat([m_r[:, -1::, :, :], m_r[:, 0:-1, :, :]], 1)

    m_n_shift = [ \
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_n_shift = tf.stack(m_n_shift, -1)

    # proxy of other_norm2
    norm2_l = tf.concat([norm2[:, :, 1::, :], norm2[:, :, 0:1, :]], 2)
    norm2_r = tf.concat([norm2[:, :, -1::, :], norm2[:, :, 0:-1, :]], 2)
    norm2_u = tf.concat([norm2[:, 1::, :, :], norm2[:, 0:1, :, :]], 1)
    norm2_d = tf.concat([norm2[:, -1::, :, :], norm2[:, 0:-1, :, :]], 1)
    norm2_lu = tf.concat([norm2_l[:, 1::, :, :], norm2_l[:, 0:1, :, :]], 1)
    norm2_ru = tf.concat([norm2_r[:, 1::, :, :], norm2_r[:, 0:1, :, :]], 1)
    norm2_ld = tf.concat([norm2_l[:, -1::, :, :], norm2_l[:, 0:-1, :, :]], 1)
    norm2_rd = tf.concat([norm2_r[:, -1::, :, :], norm2_r[:, 0:-1, :, :]], 1)
    other_norm2 = tf.stack([ \
        norm2_rd, norm2_d, norm2_ld, norm2_r,
        norm2_l, norm2_ru, norm2_u, norm2_lu,
    ], -1)

    dist = [ \
        m_rd * m_n, m_d * m_n, m_ld * m_n, m_r * m_n,
        m_l * m_n, m_ru * m_n, m_u * m_n, m_lu * m_n,
    ]
    dist = -tf.reduce_sum(tf.stack(dist, -1), -2)
    dist += 1.0
    dist *= 0.5

    # color filtering
    gauss_f = gauss.flatten()
    gauss_f = np.delete(gauss_f, [4])
    joint_bilateral_exp = tf.tile(tf.expand_dims(joint_bilateral_exp, -1), [1, 1, 1, 1, 8])
    weight_f = tf.exp(-1.442695 * joint_bilateral_exp * dist)
    weight = tf.stack([gauss_f[k] * weight_f[:, :, :, :, k] for k in range(weight_f.shape[-1])], -1)

    # if (other_norm2 >= threshold)...
    threshold = tf.tile(tf.expand_dims(threshold, -1), [1, 1, 1, 1, 8])
    wgt_msk = tf.cast(tf.less(threshold, other_norm2), dtype=dtype)
    weight = wgt_msk * weight
    dist = wgt_msk * dist

    # coefficient for bilateral space
    ms = tf.stack([m[:, :, :, 0:3], m[:, :, :, 3:6]], -1)

    # proxy for other m normalized
    m_l = tf.concat([ms[:, :, 1::, :], ms[:, :, 0:1, :]], 2)
    m_r = tf.concat([ms[:, :, -1::, :], ms[:, :, 0:-1, :]], 2)
    m_u = tf.concat([ms[:, 1::, :, :], ms[:, 0:1, :, :]], 1)
    m_d = tf.concat([ms[:, -1::, :, :], ms[:, 0:-1, :, :]], 1)
    m_lu = tf.concat([m_l[:, 1::, :, :], m_l[:, 0:1, :, :]], 1)
    m_ru = tf.concat([m_r[:, 1::, :, :], m_r[:, 0:1, :, :]], 1)
    m_ld = tf.concat([m_l[:, -1::, :, :], m_l[:, 0:-1, :, :]], 1)
    m_rd = tf.concat([m_r[:, -1::, :, :], m_r[:, 0:-1, :, :]], 1)
    m_shift = [ \
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_shift = tf.stack(m_shift, -1)

    weighted_m_acc0 += tf.reduce_sum(weight * m_shift[:, :, :, :, 0, :], -1)
    weighted_m_acc1 += tf.reduce_sum(weight * m_shift[:, :, :, :, 1, :], -1)

    dist_acc = tf.reduce_sum(dist, -1)
    weight_acc += tf.reduce_sum(weight, -1)

    # test the edge
    bilateral_max_edge_test = tf.reduce_prod(tf.cast( \
        tf.less(dist_acc, prms['joint_bilateral_max_edge']),
        dtype
    ), -1)

    m_out = []
    wgt_acc_msk = tf.cast(tf.less(0.0, weight_acc), dtype=dtype)
    m_out.append(wgt_acc_msk * weighted_m_acc0 / weight_acc)
    m_out.append(wgt_acc_msk * weighted_m_acc1 / weight_acc)
    m_out.append(m[:, :, :, 6:9])

    m_out = tf.concat(m_out, -1)

    # mask out the edge
    # do not filter the edge
    edge_step = 1
    edge_msk = np.zeros(m.shape[1:3])
    edge_msk[0:0 + edge_step, :] = 1
    edge_msk[-1 - edge_step + 1::, :] = 1
    edge_msk[:, 0:0 + edge_step] = 1
    edge_msk[:, -1 - edge_step + 1::] = 1
    edge_msk = tf.constant(edge_msk, dtype=dtype)
    edge_msk = tf.tile(tf.expand_dims(tf.expand_dims(edge_msk, -1), 0), [tf.shape(m)[0], 1, 1, 9])

    m_out = edge_msk * m + (1 - edge_msk) * m_out

    return m_out, bilateral_max_edge_test
def filterPixelStage2(raw_depth, raw_depth_edge, ir_sum):
    # raw depth is the raw depth prediction
    # raw_depth_edge is roughly the same as raw depth, except some part are zero if
    # don't want to do edge filtering
    # mask out depth that is out of region
    depth_msk = tf.cast(tf.greater(raw_depth, prms['min_depth']), dtype) * \
                tf.cast(tf.less(raw_depth, prms['max_depth']), dtype)
    # mask out the edge
    # do not filter the edge of the image
    edge_step = 1
    edge_msk = np.zeros(raw_depth.shape[1:3])
    edge_msk[0:0 + edge_step, :] = 1
    edge_msk[-1 - edge_step + 1::, :] = 1
    edge_msk[:, 0:0 + edge_step] = 1
    edge_msk[:, -1 - edge_step + 1::] = 1
    edge_msk = tf.constant(edge_msk, dtype=dtype)
    edge_msk = tf.tile(tf.expand_dims(edge_msk, 0), [tf.shape(raw_depth)[0], 1, 1])

    #
    knl = tf.constant(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), dtype=dtype)
    knl = tf.expand_dims(tf.expand_dims(knl, -1), -1)
    ir_sum_exp = tf.expand_dims(ir_sum, -1)
    ir_sum_acc = tf.nn.conv2d(ir_sum_exp, knl, strides=[1, 1, 1, 1], padding='SAME')
    squared_ir_sum_acc = tf.nn.conv2d(ir_sum_exp ** 2, knl, strides=[1, 1, 1, 1], padding='SAME')
    ir_sum_acc = tf.squeeze(ir_sum_acc, -1)
    squared_ir_sum_acc = tf.squeeze(squared_ir_sum_acc, -1)
    min_depth = raw_depth
    max_depth = raw_depth

    # min_depth, max_depth
    m_n = raw_depth_edge
    m_l = tf.concat([m_n[:, :, 1::], m_n[:, :, 0:1]], 2)
    m_r = tf.concat([m_n[:, :, -1::], m_n[:, :, 0:-1]], 2)
    m_u = tf.concat([m_n[:, 1::, :], m_n[:, 0:1, :]], 1)
    m_d = tf.concat([m_n[:, -1::, :], m_n[:, 0:-1, :]], 1)
    m_lu = tf.concat([m_l[:, 1::, :], m_l[:, 0:1, :]], 1)
    m_ru = tf.concat([m_r[:, 1::, :], m_r[:, 0:1, :]], 1)
    m_ld = tf.concat([m_l[:, -1::, :], m_l[:, 0:-1, :]], 1)
    m_rd = tf.concat([m_r[:, -1::, :], m_r[:, 0:-1, :]], 1)
    m_shift = [ \
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_shift = tf.stack(m_shift, -1)
    nonzero_msk = tf.cast(tf.greater(m_shift, 0.0), dtype=dtype)
    m_shift_min = nonzero_msk * m_shift + (1 - nonzero_msk) * 99999999999
    min_depth = tf.minimum(tf.reduce_min(m_shift_min, -1), min_depth)
    max_depth = tf.maximum(tf.reduce_max(m_shift, -1), max_depth)

    #
    tmp0 = tf.sqrt(squared_ir_sum_acc * 9.0 - ir_sum_acc ** 2) / 9.0
    edge_avg = tf.maximum( \
        ir_sum_acc / 9.0, prms['edge_ab_avg_min_value']
    )
    tmp0 /= edge_avg

    #
    abs_min_diff = tf.abs(raw_depth - min_depth)
    abs_max_diff = tf.abs(raw_depth - max_depth)

    avg_diff = (abs_min_diff + abs_max_diff) * 0.5
    max_abs_diff = tf.maximum(abs_min_diff, abs_max_diff)

    cond0 = []
    cond0.append(tf.cast(tf.less(0.0, raw_depth), dtype))
    cond0.append(tf.cast(tf.greater_equal(tmp0, prms['edge_ab_std_dev_threshold']), dtype))
    cond0.append(tf.cast(tf.less(prms['edge_close_delta_threshold'], abs_min_diff), dtype))
    cond0.append(tf.cast(tf.less(prms['edge_far_delta_threshold'], abs_max_diff), dtype))
    cond0.append(tf.cast(tf.less(prms['edge_max_delta_threshold'], max_abs_diff), dtype))
    cond0.append(tf.cast(tf.less(prms['edge_avg_delta_threshold'], avg_diff), dtype))

    cond0 = tf.reduce_prod(tf.stack(cond0, -1), -1)

    depth_out = (1 - cond0) * raw_depth

    # !cond0 part
    edge_test_msk = 1 - tf.cast(tf.equal(raw_depth_edge, 0.0), dtype)
    depth_out = raw_depth * (1 - cond0) * edge_test_msk

    # mask out the depth out of the range
    depth_out = depth_out * depth_msk

    # mask out the edge
    depth_out = edge_msk * raw_depth + (1 - edge_msk) * depth_out

    # msk_out
    msk_out = edge_msk + (1 - edge_msk) * depth_msk * (1 - cond0) * edge_test_msk

    return depth_out, msk_out