import tensorflow as tf
from functools import partial

def pad2d(x, vpad, hpad):
    return tf.pad(x, [[0, 0], vpad, hpad, [0, 0]])

def crop2d(x, vcrop, hcrop):
    return tf.keras.layers.Cropping2D([vcrop, hcrop])(x)

def get_cost(features_0, features_0from1, shift):
    """
    Calculate cost volume for specific shift

    - inputs
    features_0 (batch, h, w, nch): feature maps at time slice 0
    features_0from1 (batch, h, w, nch): feature maps at time slice 0 warped from 1
    shift (2): spatial (vertical and horizontal) shift to be considered

    - output
    cost (batch, h, w): cost volume map for the given shift
    """
    v, h = shift # vertical/horizontal element
    vt, vb, hl, hr =  max(v,0), abs(min(v,0)), max(h,0), abs(min(h,0)) # top/bottom left/right
    f_0_pad = pad2d(features_0, [vt, vb], [hl, hr])
    f_0from1_pad = pad2d(features_0from1, [vb, vt], [hr, hl])
    cost_pad = f_0_pad*f_0from1_pad
    return tf.reduce_mean(crop2d(cost_pad, [vt, vb], [hl, hr]), axis = 3)

def costvolumelayer(f1, f2, search_range=4):

    b, h, w, f = tf.unstack(tf.shape(f1))
    cost_length = (2*search_range+1)**2
    get_c = partial(get_cost, f1, f2)
    cv = [0]*cost_length
    depth=0
    for v in range(-search_range, search_range+1):
        for h in range(-search_range, search_range+1):
            cv[depth] = get_c(shift=[v,h])
            depth+=1
    cv = tf.stack(cv, axis=3)
    cv = tf.nn.leaky_relu(cv, 0.1)

    return cv

def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[: ,: ,: ,1]) # flow_y
    warped_gx = tf.add(grid_x, flow[: ,: ,: ,0]) # flow_x
    # clip value by height/width limitation
    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h- 1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w - 1)

    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis=3)

    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x


def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis=-1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0 + 1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0 + 1

    # warping indices
    h_lim = tf.cast(h - 1, tf.float32)
    w_lim = tf.cast(w - 1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)

    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis=3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis=3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis=3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis=3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy) * (fx_1 - fx), axis=3)
    c_01 = tf.expand_dims((fy_1 - fy) * (fx - fx_0), axis=3)
    c_10 = tf.expand_dims((fy - fy_0) * (fx_1 - fx), axis=3)
    c_11 = tf.expand_dims((fy - fy_0) * (fx - fx_0), axis=3)

    return c_00 * x_00 + c_01 * x_01 + c_10 * x_10 + c_11 * x_11