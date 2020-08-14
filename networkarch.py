import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import Identity

from DenseRes import DenseResBlock

class networkarch(keras.Model):
    def __init__(self,
                 n_inputs=128,
                 n_latent=20,
                 len_time = 51,
                 num_shifts=50,
                 num_shifts_middle=50,
                 outer_encoder=DenseResBlock(),
                 outer_decoder=DenseResBlock(),
                 inner_config=dict(kernel_initializer=identity_init)
                 **kwargs):
        super().__init__(**kwargs)

        self.n_inputs = n_inputs
        self.n_latent = n_latent
        self.outer_encoder = outer_encoder
        self.outer_decoder = outer_decoder
        self.inner_encoder = keras.layers.Dense(n_latent,
                                                name='inner_encoder',
                                                activation=None,
                                                use_bias=False,
                                                **inner_config)
        self.L = tf.Variable(tf.eye(n_latent), trainable=True)
        self.inner_decoder = keras.layers.Dense(n_inputs, 
                                                name='inner_decoder',
                                                activation=None,
                                                use_bias=False,
                                                **inner_config)

        self.rel_mse = rel_mse(name='relative_mse')
        self.train_autoencoders_only

        def call(self, inputs):
            uk = inputs

            # Autoencoder
            partially_encoded = self.outer_encoder(uk)
            vk = self.inner_encoder(outer_encoded)
            vkplus1 = tf.matmul(vk, self.L)
            partially_decoded = self.inner_decoder(vkplus1)
            ukplus1 = self.outer_decoder(partially_decoded)

            # Prediction - do some stacking


            # Linearity - do some stacking

            return reconstructed_x, predictions, outer_reconst_x


# Losses:
    # Outputs:
    # Autoencoder    
    # Prediction
    # Outer autoencoder

    # Model Internals:
    # Linearity loss
    # Inner autoencoder        

def identity_init(shape, dtype=tf.float32):
    n_rows = shape[0]
    n_cols = shape[1]
    if n_rows >= n_cols:
        A = np.zeros((n_rows,n_cols), dtype=np.float32)
        for col in range(n_cols):
            for row in range(col,col+n_rows-n_cols+1):
                A[row,col] = 1.0/(n_rows-n_cols+1) 
    else:
        A = np.zeros((n_rows,n_cols), dtype=np.float32)
        for row in range(n_rows):
            for col in range(row,row+n_cols-n_rows+1):
                A[row,col] = 1.0/(n_cols-n_rows+1) 
    return A


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

def outer_encoder_decoder_fc(
    input_width, num_hidden_layers, hidden_width, act_type, initialization, 
    L1_lam, L2_lam, add_identity
):
    denselayer = partial(
        layers.Dense, activation=act_type, kernel_initializer=initialization,
        kernel_regularizer=keras.regularizers.l1_l2(l1=L1_lam,l2=L2_lam)
    )
    input_layer = layers.Input(shape=[input_width])
    x = denselayer(hidden_width)(input_layer)
    for i in np.arange(num_hidden_layers-1):
        x = denselayer(hidden_width)(x)
    linear_layer = denselayer(input_width, activation=None)(x)
    if add_identity:
        resnet_layer = layers.add([linear_layer,input_layer])
    else:
        resnet_layer = linear_layer
    return keras.Model(input_layer, resnet_layer)

def create_koopman_fcnet(params):
    max_shifts_to_stack = helperfns.num_shifts_in_stack(params)

    x = tf.placeholder(tf.float32, shape=[max_shifts_to_stack + 1, None, params['widths'][0]], name="x")

    reg = keras.regularizers.l1_l2(l1=params['L1_lam'],l2=params['L1_lam'])
    matmullayer = partial(
        layers.Dense, activation=None, use_bias=False
        kernel_initializer=identity_init, kernel_regularizer=reg
    )

    # Create the different parts of the network
    outer_encoder = outer_encoder_decoder_fc(
        input_width=params['input_width'], 
        num_hidden_layers=params['num_hidden_layers'], 
        hidden_width=params['hidden_width'], 
        act_type=params['act_type'], initialization=params['initialization'], 
        L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], 
        add_identity=params['add_identity']
    )
    inner_encoder = matmullayer(params['num_evals'], name='inner_encoder')
    L = tf.Variable(tf.eye(params['num_evals']), trainable=True, 
                    dtype=tf.float32, name='L')
    inner_decoder = matmullayer(params['output_width'], name='inner_decoder')
    outer_decoder = outer_encoder_decoder_fc(
        input_width=params['output_width'],
        num_hidden_layer=params['num_hidden_layers'],
        hidden_width=params['hidden_width'],
        act_type=params['act_type'], initialization=params['initialization'],
        L1_lam=params['L1_lam'], L2_lam=params['L2_lam'], 
        add_identity=params['add_identity']
    )

    # Feed the inputs through the network
    x = layers.Input(shape=[params['input_width']], name='x')
    partially_encoded = outer_encoder(x)
    encoded = inner_encoder(partially_encoded)
    partially_decoded_x = inner_decoder(encoded)
    reconstructed_x = outer_decoder(partially_decoded_x)
    outer_reconstructed_x = outer_decoder(partially_encoded)

    advanced_layer = tf.identity(encoded)
    y = []
    advanced_encoded = []
    for i in range(max_shifts_to_stack):
        advanced_layer = tf.matmul(advanced_layer,L)
        advanced_part_decoded = inner_decoder(advanced_layer)
        advanced_decoded = outer_decoder(advanced_part_decoded)
        if i < params['num_shifts']:
            y.append(advanced_decoded)
        if i < params['num_shifts_middle']:
            advanced_encoded.append(advanced_layer)

    Koopman_Autoencoder = keras.Model(
        inputs=x, outputs=[y,reconstructed_x,outer_reconstructed_x]
    )

    # Add inner autoencoder loss
    Koopman_Autoecoder.add_loss(calculate_loss(
                                partially_encoded, partially_decoded,
                                denominator_nonzero=1e-5,
                                rel_loss=params['relative_loss'],
                                lam=params['inner_autoencoder_loss_lam']
                                ))

    # Add linearity loss
    


    # Also do prediction and linearity loss here???

    # x is the stacked data
    # y  is the predictions using the network
    # partial_encoded_list is x partially encoded
    # g_list is x fully encoded
    # reconstructed_x is x through autoencoder (no L)
    # outer reconstructed_x is x through outer autoencoder (no L)

    # Autoencoder Loss: x vs reconstructed x
    # Prediction Loss: x vs y - with some shifting/gathering???
    # Linearity Loss: next step in g_list vs L*g_list
    # Inner auto: partially encoded vs partially decoded
    # Outer auto: x vs outer_reconstructed_x

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