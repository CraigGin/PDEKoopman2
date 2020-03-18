import numpy as np
import tensorflow as tf
from scipy.linalg import dft
from functools import partial

import helperfns

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if len(shape) == 1:
            return tf.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2:
            return tf.constant(helperfns.identity_seed(shape[0],shape[1]))
        elif len(shape) == 3:
            array = np.zeros(shape, dtype=np.float32)
            for i in range(shape[2]):
                std_dev = 0.1/(np.abs(shape[1]-shape[0])+1)
                array[:, :, i] = helperfns.identity_seed(shape[0],shape[1])+np.random.normal(0,std_dev,(shape[0],shape[1]))
            return tf.constant(array)
        else:
            raise ValueError("Error, initializer expected a different shape: " % len(shape))
    return _initializer


def create_FT_layer(n_inputs, n_middle, seed_middle, fix_middle, L1_lam, L2_lam):
    if not seed_middle:
        FT = tf.get_variable("FT", initializer=helperfns.identity_seed(n_inputs,n_middle), 
                            regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            trainable=True, dtype=tf.float32)        
    else:
        if not fix_middle:
            FT = tf.get_variable("FT", initializer=helperfns.reduced_DFT(n_inputs,n_middle),
                            regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            trainable=True, dtype=tf.float32)
        else:
            FT = tf.get_variable("FT", initializer=helperfns.reduced_DFT(n_inputs,n_middle),
                            regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            trainable=False, dtype=tf.float32)

    return FT   

def create_IFT_layer(n_middle, n_outputs, seed_middle, fix_middle, L1_lam, L2_lam):
    if not seed_middle:
        IFT = tf.get_variable("IFT", initializer=helperfns.identity_seed(n_middle,n_outputs), 
                            regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            trainable=True, dtype=tf.float32)  
    else:
        if not fix_middle:
            IFT = tf.get_variable("IFT", initializer=helperfns.expand_IDFT(n_middle,n_outputs),
                            regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            trainable=True, dtype=tf.float32)
        else:
            IFT = tf.get_variable("IFT", initializer=helperfns.expand_IDFT(n_middle,n_outputs),
                            regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            trainable=False, dtype=tf.float32)

    return IFT  

def encoder_apply_cn(x, n_inputs, conv1_filters, n_middle, L1_lam, L2_lam, shifts_middle, num_shifts_max, fix_middle, seed_middle, add_identity, initialization):
    partially_encoded_list = []
    encoded_list = []
    num_shifts_middle = len(shifts_middle)
    for j in np.arange(num_shifts_max + 1):
        if j == 0:
            shift = 0
            reuse = False
        else:
            shift = shifts_middle[j - 1]
            reuse = True
        if isinstance(x, (list,)):
            x_shift = x[shift]
        else:
            x_shift = tf.squeeze(x[shift, :, :])
        partially_encoded, encoded = encoder_apply_one_shift_cn(x_shift, n_inputs, conv1_filters, n_middle, L1_lam, L2_lam, reuse, 
                                                            fix_middle, seed_middle, add_identity, initialization)
        partially_encoded_list.append(partially_encoded)
        if j <= num_shifts_middle:
            encoded_list.append(encoded)

    return partially_encoded_list, encoded_list


def encoder_apply_one_shift_cn(x, n_inputs, conv1_filters, n_middle, L1_lam, L2_lam, reuse, fix_middle, seed_middle, add_identity, initialization):

    my_conv_layer = partial(tf.layers.conv1d, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                             bias_regularizer=None)
    my_dense_layer = partial(tf.layers.dense, activation=tf.nn.relu, kernel_initializer=initialization,
                            kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            bias_regularizer=None)
    my_linear_layer = partial(tf.layers.dense, activation=None, kernel_initializer=initialization,
                            kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            bias_regularizer=None)                   

    with tf.variable_scope("encoder", reuse=reuse):
        uk_reshaped = tf.reshape(x, shape=[-1, n_inputs, 1], name="uk_reshaped")
        hidden1_encode = my_conv_layer(uk_reshaped, filters=8, kernel_size=4, strides=1, padding="SAME",
                                name="hidden1_encode")
        hidden2_encode = tf.layers.average_pooling1d(hidden1_encode, pool_size=2, strides=2, padding="VALID", name="hidden2_encode")
        hidden3_encode = my_conv_layer(hidden2_encode, filters=16, kernel_size=4, strides=1, padding="SAME",
                                name="hidden3_encode")
        hidden4_encode = tf.layers.average_pooling1d(hidden3_encode, pool_size=2, strides=2, padding="VALID", name="hidden4_encode")
        hidden5_encode = my_conv_layer(hidden4_encode, filters=32, kernel_size=4, strides=1, padding="SAME",
                                name="hidden5_encode")
        hidden6_encode = tf.layers.average_pooling1d(hidden5_encode, pool_size=2, strides=2, padding="VALID", name="hidden6_encode")
        hidden7_encode = my_conv_layer(hidden6_encode, filters=64, kernel_size=4, strides=1, padding="SAME", 
                                 name="hidden7_encode")
        hidden7_encode_reshaped = tf.reshape(hidden7_encode, shape=[-1, 16*64], name="hidden7_encode_reshaped")
        hidden8_encode = my_dense_layer(hidden7_encode_reshaped, n_inputs, name="hidden8_encode")
        hidden9_encode = my_linear_layer(hidden8_encode, n_inputs, name="hidden9_encode")

        if add_identity:
            identity_weight = tf.get_variable(name='alphaE', dtype=np.float32, initializer=tf.constant(add_identity, dtype=tf.float32), trainable=False)
        else:
            identity_weight = 0
        partially_encoded = tf.add(hidden9_encode,tf.scalar_mul(identity_weight, x), name="v_k")

        FT = create_FT_layer(n_inputs, n_middle, seed_middle, fix_middle, L1_lam, L2_lam)
        encoded = tf.matmul(partially_encoded,FT, name="vk_hat")

    return partially_encoded, encoded

def decoder_apply_cn(x, n_middle, conv2_filters, n_outputs, L1_lam, L2_lam, reuse, fix_middle, seed_middle, add_identity, initialization):
    prev_layer = tf.identity(x)

    with tf.variable_scope("decoder_inner", reuse=reuse):
        IFT = create_IFT_layer(n_middle, n_outputs, seed_middle, fix_middle, L1_lam, L2_lam)
        prev_layer = tf.matmul(prev_layer, IFT, name="vkplus1") 
        
    output = outer_decoder_apply_cn(prev_layer, conv2_filters, n_outputs, L1_lam, L2_lam, reuse, add_identity, initialization)

    return output

def outer_decoder_apply_cn(x, conv2_filters, n_outputs, L1_lam, L2_lam, reuse, add_identity, initialization):

    my_conv_layer = partial(tf.layers.conv1d, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                             bias_regularizer=None)
    my_dense_layer = partial(tf.layers.dense, activation=tf.nn.relu, kernel_initializer=initialization,
                            kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            bias_regularizer=None)
    my_linear_layer = partial(tf.layers.dense, activation=None, kernel_initializer=initialization,
                            kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            bias_regularizer=None)   

    with tf.variable_scope("decoder_outer", reuse=reuse):
        vkplus1_reshaped = tf.reshape(x, shape=[-1, n_outputs, 1], name="vkplus1_reshaped")
        hidden1_decode = my_conv_layer(vkplus1_reshaped, filters=8, kernel_size=4, strides=1, padding="SAME",
                                name="hidden1_decode")
        hidden2_decode = tf.layers.average_pooling1d(hidden1_decode, pool_size=2, strides=2, padding="VALID", name="hidden2_decode")
        hidden3_decode = my_conv_layer(hidden2_decode, filters=16, kernel_size=4, strides=1, padding="SAME",
                                name="hidden3_decode")
        hidden4_decode = tf.layers.average_pooling1d(hidden3_decode, pool_size=2, strides=2, padding="VALID", name="hidden4_decode")
        hidden5_decode = my_conv_layer(hidden4_decode, filters=32, kernel_size=4, strides=1, padding="SAME",
                                name="hidden5_decode")
        hidden6_decode = tf.layers.average_pooling1d(hidden5_decode, pool_size=2, strides=2, padding="VALID", name="hidden6_decode")
        hidden7_decode = my_conv_layer(hidden6_decode, filters=64, kernel_size=4, strides=1, padding="SAME", 
                                 name="hidden7_decode")
        hidden7_decode_reshaped = tf.reshape(hidden7_decode, shape=[-1, 16*64], name="hidden7_decode_reshaped")
        hidden8_decode = my_dense_layer(hidden7_decode_reshaped, n_outputs, name="hidden8_decode")
        hidden9_decode = my_linear_layer(hidden8_decode, n_outputs, name="hidden9_decode")

        if add_identity:
            identity_weight = tf.get_variable(name='alphaD', dtype=np.float32, initializer=tf.constant(add_identity, dtype=tf.float32), trainable=False)
        else:
            identity_weight = 0

        output = tf.add(hidden9_decode,tf.scalar_mul(identity_weight, x), name="outputs")

    return output
    
def create_koopman_convnet(params):
    max_shifts_to_stack = helperfns.num_shifts_in_stack(params)

    if params['initialization'] == 'identity':
        initialization = identity_initializer()
    elif params['initialization'] == 'He':
        initialization = tf.contrib.layers.variance_scaling_initializer()
    else: 
        raise ValueError("Error, initialization must be either identity or He")

    n_inputs = params['n_inputs']
    conv1_filters = params['conv1_filters']
    n_middle = params['n_middle']
    conv2_filters = params['conv2_filters']
    n_outputs = params['n_outputs']

    x = tf.placeholder(tf.float32, shape=[max_shifts_to_stack + 1, None, n_inputs], name="x")

    # returns list: encode each shift
    partial_encoded_list, g_list = encoder_apply_cn(x, n_inputs, conv1_filters, n_middle, L1_lam=params['L1_lam'], L2_lam=params['L2_lam'],
                                                shifts_middle=params['shifts_middle'], num_shifts_max=max_shifts_to_stack, 
                                                fix_middle=params['fix_middle'], seed_middle=params['seed_middle'], 
                                                add_identity=params['add_identity'], initialization=initialization)

    if not params['seed_middle']:
        if not params['diag_L']:
            with tf.variable_scope("dynamics", reuse=False):
                L = tf.get_variable("L", shape=[n_middle,n_middle], initializer=identity_initializer(), trainable=True, dtype=tf.float32)
        else:
            with tf.variable_scope("dynamics", reuse=False):
                diag = tf.get_variable("diag", initializer=np.ones(n_middle, dtype=np.float32), trainable=True, dtype=tf.float32)
                L = tf.diag(diag, name="L")
    else:
        # Fix/seed middle as heat equation
        kv = helperfns.freq_vector(n_middle)
        dt = params['delta_t']
        if not params['fix_middle']:
            if not params['diag_L']:
                with tf.variable_scope("dynamics", reuse=False):
                    L = tf.get_variable("L", initializer=np.float32(np.diag(np.exp(-params['mu']*kv*kv*dt))), trainable=True, dtype=tf.float32)
            else:
                with tf.variable_scope("dynamics", reuse=False):
                    diag = tf.get_variable("diag", initializer=np.float32(np.exp(-params['mu']*kv*kv*dt)), trainable=True, dtype=tf.float32)
                    L = tf.diag(diag, name="L")
        else:
            with tf.variable_scope("dynamics", reuse=False):
                mu = tf.get_variable("mu", shape=[1], initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True, dtype=tf.float32)
                L = tf.diag(tf.exp(-mu*kv*kv*dt), name="L")

    y = []
    # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
    encoded_layer = g_list[0]

    y.append(decoder_apply_cn(encoded_layer, n_middle, conv2_filters, n_outputs, L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], reuse=False, 
                            fix_middle=params['fix_middle'], seed_middle=params['seed_middle'], add_identity=params['add_identity'], 
                            initialization=initialization))

    reconstructed_x = []
    for j in np.arange(max_shifts_to_stack + 1):
        reconstructed_x.append(decoder_apply_cn(g_list[j], n_middle, conv2_filters, n_outputs, L1_lam=params['L1_lam'], L2_lam=params['L2_lam'],                           
                            reuse=True, fix_middle=params['fix_middle'], seed_middle=params['seed_middle'], add_identity=params['add_identity'],
                            initialization=initialization))

    outer_reconst_x = []
    for j in np.arange(max_shifts_to_stack + 1):
        outer_reconst_x.append(outer_decoder_apply_cn(partial_encoded_list[j], conv2_filters, n_outputs, 
                                L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], reuse=True, add_identity=params['add_identity'],
                                initialization=initialization))

    if not params['autoencoder_only']:
        # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
        advanced_layer = tf.matmul(encoded_layer, L)

        for j in np.arange(max(params['shifts'])):  # loops 0, 1, ...
            # considering penalty on subset of yk+1, yk+2, yk+3, ... yk+20
            if (j + 1) in params['shifts']:
                y.append(decoder_apply_cn(advanced_layer, n_middle, conv2_filters, n_outputs, L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], 
                                        reuse=True, fix_middle=params['fix_middle'], seed_middle=params['seed_middle'],
                                        add_identity=params['add_identity'], initialization=initialization))

            advanced_layer = tf.matmul(advanced_layer, L)

    if len(y) != (len(params['shifts']) + 1):
        print "messed up looping over shifts! %r" % params['shifts']
        raise ValueError(
            'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

    return x, y, partial_encoded_list, g_list, reconstructed_x, outer_reconst_x

def encoder_apply_fc(x, widths, linear_encoder_layers, act_type, log_space, L1_lam, L2_lam, shifts_middle, num_shifts_max, 
                    fix_middle, seed_middle, add_identity, num_encoder_weights, initialization):
    partially_encoded_list = []
    encoded_list = []
    num_shifts_middle = len(shifts_middle)
    for j in np.arange(num_shifts_max + 1):
        if j == 0:
            shift = 0
            reuse = False
        else:
            shift = shifts_middle[j - 1]
            reuse = True
        if isinstance(x, (list,)):
            x_shift = x[shift]
        else:
            x_shift = tf.squeeze(x[shift, :, :])
        partially_encoded, encoded = encoder_apply_one_shift_fc(x_shift, widths, act_type, log_space, L1_lam, L2_lam, reuse, fix_middle, 
                                                            seed_middle, add_identity, num_encoder_weights, linear_encoder_layers, initialization)
        partially_encoded_list.append(partially_encoded)
        if j <= num_shifts_middle:
            encoded_list.append(encoded)

    return partially_encoded_list, encoded_list

def encoder_apply_one_shift_fc(x, widths, act_type, log_space, L1_lam, L2_lam, reuse, fix_middle, seed_middle, add_identity, 
                            num_encoder_weights, linear_encoder_layers, initialization):
    prev_layer = tf.identity(x)

    my_dense_layer = partial(tf.layers.dense, activation=act_type, kernel_initializer=initialization,
                            kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            bias_regularizer=None)
    my_linear_layer = partial(tf.layers.dense, activation=None, kernel_initializer=initialization,
                            kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                            bias_regularizer=None)

    with tf.variable_scope("encoder", reuse=reuse):
        if not log_space:
            prev_layer = tf.identity(prev_layer, name="log_uk")
        else:
            prev_layer = tf.log(prev_layer+2, name="log_uk")
        
        for i in np.arange(num_encoder_weights - 1):
            if i not in linear_encoder_layers:
                prev_layer = my_dense_layer(tf.reshape(prev_layer,[-1,widths[i]]), widths[i+1], name="hidden%d_encode" % (i + 1))
            else:
                prev_layer = my_linear_layer(tf.reshape(prev_layer,[-1,widths[i]]), widths[i+1], name="hidden%d_encode" % (i + 1))

        if add_identity:
            identity_weight = tf.get_variable(name='alphaE', dtype=np.float32, initializer=tf.constant(add_identity, dtype=tf.float32), trainable=False)
        else:
            identity_weight = 0
        add_iden = tf.add(prev_layer,tf.scalar_mul(identity_weight, x))

        if not log_space:
            partially_encoded = tf.identity(add_iden, name="v_k")
        else:
            partially_encoded  = tf.exp(add_iden, name="v_k")

        FT = create_FT_layer(widths[num_encoder_weights-1], widths[num_encoder_weights], seed_middle, fix_middle, L1_lam, L2_lam)
        encoded = tf.matmul(partially_encoded,FT, name="vk_hat")

    return partially_encoded, encoded

def decoder_apply_fc(x, widths, linear_decoder_layers, act_type, log_space, L1_lam, L2_lam, reuse,
                    fix_middle, seed_middle, add_identity, num_decoder_weights, initialization):
    prev_layer = tf.identity(x)

    with tf.variable_scope("decoder_inner", reuse=reuse):
        IFT = create_IFT_layer(widths[-(num_decoder_weights+1)], widths[-1], seed_middle, fix_middle, L1_lam, L2_lam)
        prev_layer = tf.matmul(prev_layer, IFT) 
        
    output = outer_decoder_apply_fc(prev_layer, widths, linear_decoder_layers, act_type, log_space, L1_lam, L2_lam, reuse, 
                                    add_identity, num_decoder_weights, initialization)

    return output

def outer_decoder_apply_fc(x, widths, linear_decoder_layers, act_type, log_space, L1_lam, L2_lam, reuse,  
                           add_identity, num_decoder_weights, initialization):
    prev_layer = tf.identity(x)

    my_dense_layer = partial(tf.layers.dense, activation=act_type, kernel_initializer=initialization,
                             kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                             bias_regularizer=None)
    my_linear_layer = partial(tf.layers.dense, activation=None, kernel_initializer=initialization,
                             kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=L1_lam,scale_l2=L2_lam), 
                             bias_regularizer=None)

    with tf.variable_scope("decoder_outer", reuse=reuse):
        if not log_space:
            prev_layer = tf.identity(prev_layer, name="log_vkplus1")
        else:
            prev_layer = tf.log(prev_layer, name="log_vkplus1")

        for i in np.arange(num_decoder_weights - 1):
            widths_i = num_decoder_weights + 2 + i
            if i+1 not in linear_decoder_layers:
                prev_layer = my_dense_layer(tf.reshape(prev_layer,[-1,widths[widths_i]]), widths[widths_i+1], name="hidden%d_decode" % (i + 1))
            else:
                prev_layer = my_linear_layer(tf.reshape(prev_layer,[-1,widths[widths_i]]), widths[widths_i+1], name="hidden%d_decode" % (i + 1))

        if add_identity:
            identity_weight = tf.get_variable(name='alphaD', dtype=np.float32, initializer=tf.constant(add_identity, dtype=tf.float32), trainable=False)
        else:
            identity_weight = 0
        output = tf.add(prev_layer,tf.scalar_mul(identity_weight, x), name="outputs")

    return output

def create_koopman_fcnet(params):
    max_shifts_to_stack = helperfns.num_shifts_in_stack(params)

    if params['initialization'] == 'identity':
        initialization = identity_initializer()
    elif params['initialization'] == 'He':
        initialization = tf.contrib.layers.variance_scaling_initializer()
    else: 
        raise ValueError("Error, initialization must be either identity or He")

    x = tf.placeholder(tf.float32, shape=[max_shifts_to_stack + 1, None, params['widths'][0]], name="x")

    # returns list: encode each shift
    partial_encoded_list, g_list = encoder_apply_fc(x, widths=params['widths'], linear_encoder_layers=params['linear_encoder_layers'], 
                                                act_type=params['act_type'], log_space=params['log_space'], 
                                                L1_lam=params['L1_lam'], L2_lam=params['L2_lam'],
                                                shifts_middle=params['shifts_middle'], num_shifts_max=max_shifts_to_stack, 
                                                fix_middle=params['fix_middle'], seed_middle=params['seed_middle'], 
                                                add_identity=params['add_identity'], num_encoder_weights=params['num_encoder_weights'],
                                                initialization=initialization)

    n_middle = params['num_evals']
    if not params['seed_middle']:
        if not params['diag_L']:
            with tf.variable_scope("dynamics", reuse=False):
                L = tf.get_variable("L", shape=[n_middle,n_middle], initializer=identity_initializer(), trainable=True, dtype=tf.float32)
        else:
            with tf.variable_scope("dynamics", reuse=False):
                diag = tf.get_variable("diag", initializer=np.ones(n_middle, dtype=np.float32), trainable=True, dtype=tf.float32)
                L = tf.diag(diag, name="L")
    else:
        # Fix/seed middle as heat equation
        kv = helperfns.freq_vector(n_middle)
        dt = params['delta_t']
        if not params['fix_middle']:
            if not params['diag_L']:
                with tf.variable_scope("dynamics", reuse=False):
                    L = tf.get_variable("L", initializer=np.float32(np.diag(np.exp(-params['mu']*kv*kv*dt))), trainable=True, dtype=tf.float32)
            else:
                with tf.variable_scope("dynamics", reuse=False):
                    diag = tf.get_variable("diag", initializer=np.float32(np.exp(-params['mu']*kv*kv*dt)), trainable=True, dtype=tf.float32)
                    L = tf.diag(diag, name="L")
        else:
            with tf.variable_scope("dynamics", reuse=False):
                mu = tf.get_variable("mu", shape=[1], initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True, dtype=tf.float32)
                L = tf.diag(tf.exp(-mu*kv*kv*dt), name="L")

    y = []
    # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
    encoded_layer = g_list[0]

    y.append(decoder_apply_fc(encoded_layer, widths=params['widths'], linear_decoder_layers=params['linear_decoder_layers'], 
                              act_type=params['act_type'], log_space=params['log_space'], 
                              L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], reuse=False, 
                              fix_middle=params['fix_middle'], seed_middle=params['seed_middle'], 
                              add_identity=params['add_identity'], num_decoder_weights=params['num_decoder_weights'],
                              initialization=initialization))

    reconstructed_x = []

    for j in np.arange(max_shifts_to_stack + 1):
        reconstructed_x.append(decoder_apply_fc(g_list[j], widths=params['widths'], linear_decoder_layers=params['linear_decoder_layers'], 
                              act_type=params['act_type'], log_space=params['log_space'], 
                              L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], reuse=True, 
                              fix_middle=params['fix_middle'], seed_middle=params['seed_middle'], 
                              add_identity=params['add_identity'], num_decoder_weights=params['num_decoder_weights'],
                              initialization=initialization))

    outer_reconst_x = []
    for j in np.arange(max_shifts_to_stack + 1):
        outer_reconst_x.append(outer_decoder_apply_fc(partial_encoded_list[j], widths=params['widths'], 
                              linear_decoder_layers=params['linear_decoder_layers'], act_type=params['act_type'], 
                              log_space=params['log_space'], L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], reuse=True, 
                              add_identity=params['add_identity'], num_decoder_weights=params['num_decoder_weights'],
                              initialization=initialization))

    if not params['autoencoder_only']:
        # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
        advanced_layer = tf.matmul(encoded_layer, L)

        for j in np.arange(max(params['shifts'])):  # loops 0, 1, ...
            # considering penalty on subset of yk+1, yk+2, yk+3, ... yk+20
            if (j + 1) in params['shifts']:
                y.append(decoder_apply_fc(advanced_layer, widths=params['widths'], linear_decoder_layers=params['linear_decoder_layers'], 
                              act_type=params['act_type'], log_space=params['log_space'], 
                              L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], reuse=True, 
                              fix_middle=params['fix_middle'], seed_middle=params['seed_middle'], 
                              add_identity=params['add_identity'], num_decoder_weights=params['num_decoder_weights'],
                              initialization=initialization))

            advanced_layer = tf.matmul(advanced_layer, L)

    if len(y) != (len(params['shifts']) + 1):
        print "messed up looping over shifts! %r" % params['shifts']
        raise ValueError(
            'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

    return x, y, partial_encoded_list, g_list, reconstructed_x, outer_reconst_x