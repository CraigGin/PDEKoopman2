import os
import time
import datetime

import numpy as np
import tensorflow as tf

import helperfns
import networkarch as net


def define_loss(x, y, partial_encoded_list, g_list, reconstructed_x, outer_reconst_x, params):
    # Minimize the mean squared errors.
    # subtraction and squaring element-wise, then average over both dimensions
    # n columns
    # average of each row (across columns), then average the rows
    with tf.variable_scope("dynamics", reuse=True):
        if not params['fix_middle']:
            if not params['diag_L']:
                L = tf.get_variable("L")
            else:
                diag = tf.get_variable("diag")
                L = tf.diag(diag, name="L")
        else:
            kv = helperfns.freq_vector(params['n_middle'])
            dt = params['delta_t']
            mu = tf.get_variable("mu")
            L = tf.diag(tf.exp(-mu*kv*kv*dt), name="L")
        
    with tf.variable_scope("encoder", reuse=True):
        FT = tf.get_variable("FT")

    with tf.variable_scope("decoder_inner", reuse=True):
        IFT = tf.get_variable("IFT")

    denominator_nonzero = 10 ** (-5)

    # autoencoder loss
    if params['autoencoder_loss_lam']:
        exact = x
        pred = reconstructed_x
        if params['relative_loss']:
            loss1_denominator = tf.reduce_mean(tf.square(exact),2) + denominator_nonzero
        else:
            loss1_denominator = tf.to_double(1.0)
        norm_squared = tf.reduce_mean(tf.square(exact - pred), 2)
        rel_error = tf.truediv(norm_squared, loss1_denominator)
        mse = tf.reduce_mean(rel_error)
        loss1 = tf.multiply(params['autoencoder_loss_lam'],mse, name="loss1")
    else:
        loss1 = tf.zeros([], dtype=tf.float32, name="loss1")

    # gets dynamics (prediction loss)
    if params['prediction_loss_lam']:
        exact = tf.gather(x,params['shifts'])
        pred = [y[i] for i in params['shifts']]
        if params['relative_loss']:
            loss2_denominator = tf.reduce_mean(tf.square(exact),2) + denominator_nonzero
        else:
            loss2_denominator = tf.to_double(1.0)
        norm_squared = tf.reduce_mean(tf.square(exact - pred), 2)
        rel_error = tf.truediv(norm_squared, loss2_denominator)
        mse = tf.reduce_mean(rel_error)
        loss2 = tf.multiply(params['prediction_loss_lam'],mse, name="loss2")
    else:
        loss2 = tf.zeros([], dtype=tf.float32, name="loss2")

    # K linear
    loss3 = tf.zeros([], dtype=tf.float32)
    if params['linearity_loss_lam']:
        count_shifts_middle = 0
        next_step = tf.matmul(g_list[0], L)
        for j in np.arange(max(params['shifts_middle'])):
            if (j + 1) in params['shifts_middle']:
                if params['relative_loss']:
                    loss3_denominator = tf.reduce_mean(tf.square(g_list[count_shifts_middle + 1]), 1) + denominator_nonzero
                else:
                    loss3_denominator = tf.to_double(1.0)
                norm_squared = tf.reduce_mean(tf.square(next_step - g_list[count_shifts_middle + 1]), 1)
                rel_error = tf.truediv(norm_squared,loss3_denominator)
                loss3 = loss3 + params['linearity_loss_lam'] * tf.reduce_mean(rel_error)
                count_shifts_middle += 1
            
            next_step = tf.matmul(next_step, L)
        loss3 = tf.truediv(loss3,tf.cast(params['num_shifts_middle'], tf.float32), name="loss3")

    # inner-autoencoder loss
    if params['inner_autoencoder_loss_lam']:
        exact = partial_encoded_list
        encoded = tf.tensordot(partial_encoded_list,FT, axes=([2],[0]))
        pred = tf.tensordot(encoded,IFT, axes=([2],[0]))
        if params['relative_loss']:
            loss4_denominator = tf.reduce_mean(tf.square(exact),2) + denominator_nonzero
        else:
            loss4_denominator = tf.to_double(1.0)
        norm_squared = tf.reduce_mean(tf.square(exact - pred), 2)
        rel_error = tf.truediv(norm_squared, loss4_denominator)
        mse = tf.reduce_mean(rel_error)
        loss4 = tf.multiply(params['inner_autoencoder_loss_lam'],mse, name="loss4")
    else:
        loss4 = tf.zeros([], dtype=tf.float32, name="loss4")

    # outer-autoencoder loss
    if params['outer_autoencoder_loss_lam']:
        exact = x
        pred = outer_reconst_x 
        if params['relative_loss']:
            loss5_denominator = tf.reduce_mean(tf.square(exact),2) + denominator_nonzero
        else:
            loss5_denominator = tf.to_double(1.0)
        norm_squared = tf.reduce_mean(tf.square(exact - pred), 2)
        rel_error = tf.truediv(norm_squared, loss5_denominator)
        mse = tf.reduce_mean(rel_error)
        loss5 = tf.multiply(params['outer_autoencoder_loss_lam'],mse, name="loss5")
    else:
        loss5 = tf.zeros([], dtype=tf.float32, name="loss5")        

    loss_list = [loss1, loss2, loss3, loss4, loss5]
    loss = tf.add_n(loss_list, name="loss")

    return loss1, loss2, loss3, loss4, loss5, loss

def define_regularization(params, loss, loss1, loss4, loss5):
    # tf.nn.l2_loss returns number
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss= tf.add_n(reg_losses)
    
    regularized_loss = loss + reg_loss
    regularized_loss1 = loss1 + reg_loss + loss4 + loss5

    return reg_loss, regularized_loss, regularized_loss1


def try_net(data_val, params):
    # SET UP NETWORK
    if params['network_arch'] == 'convnet':
        x, y, partial_encoded_list, g_list, reconstructed_x, outer_reconst_x = net.create_koopman_convnet(params)
    elif params['network_arch'] == 'fully_connected':
        x, y, partial_encoded_list, g_list, reconstructed_x, outer_reconst_x = net.create_koopman_fcnet(params)
    else: 
        raise ValueError("Error, network_arch must be either convnet or fully_connected")

    max_shifts_to_stack = helperfns.num_shifts_in_stack(params)

    # DEFINE LOSS FUNCTION
    trainable_var = tf.trainable_variables()
    loss1, loss2, loss3, loss4, loss5, loss = define_loss(x, y, partial_encoded_list, g_list,
                                                                     reconstructed_x, outer_reconst_x, params)
    reg_loss, regularized_loss, regularized_loss1 = define_regularization(params, loss, loss1, loss4, loss5)
    losses = {'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss4': loss4, 'loss5': loss5,
              'loss': loss, 'reg_loss': reg_loss, 'regularized_loss': regularized_loss,
              'regularized_loss1': regularized_loss1}

    # CHOOSE OPTIMIZATION ALGORITHM
    optimizer = helperfns.choose_optimizer(params, regularized_loss, trainable_var)
    optimizer_autoencoder = helperfns.choose_optimizer(params, regularized_loss1, trainable_var)

    # LAUNCH GRAPH AND INITIALIZE
    # Use specified fraction of GPU Memory
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # Start with only as much GPU usage as needed and allow it to grow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Use all of GPU memory
    # sess = tf.Session()

    saver = tf.train.Saver()

    # Before starting, initialize the variables.  
    if not params['restore']:
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        saver.restore(sess, params['model_restore_path'])
        params['exp_suffix'] = '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        exp_name = params['data_name'] + params['exp_suffix']
        params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], exp_name)

    csv_path = params['model_path'].replace('model', 'error')
    csv_path = csv_path.replace('ckpt', 'csv')
    print csv_path

    num_saved_per_file_pass = params['num_steps_per_file_pass'] / 20 + 1
    num_saved = np.floor(num_saved_per_file_pass * params['data_train_len'] * params['num_passes_per_file']).astype(int)
    train_val_error = np.zeros([num_saved, 16])
    count = 0
    best_error = 10000

    data_val_tensor = helperfns.stack_data(data_val, max_shifts_to_stack, params['val_len_time'])

    start = time.time()
    finished = 0
    saver.save(sess, params['model_path'])

    # TRAINING
    # loop over training data files
    for f in xrange(params['data_train_len'] * params['num_passes_per_file']):
        if finished:
            break
        file_num = (f % params['data_train_len']) + 1  # 1...data_train_len

        if (params['data_train_len'] > 1) or (f == 0):
            # don't keep reloading data if always same
            data_train = np.load(('./data/%s_train%d_x.npy' % (params['data_name'], file_num)))
            data_train_tensor = helperfns.stack_data(data_train, max_shifts_to_stack,
                                                     params['train_len_time'][file_num - 1])
            num_examples = data_train_tensor.shape[1]
            num_batches = int(np.floor(num_examples / params['batch_size']))
        ind = np.arange(num_examples)
        np.random.shuffle(ind)
        data_train_tensor = data_train_tensor[:, ind, :]

        # loop over batches in this file
        for step in xrange(params['num_steps_per_batch'] * num_batches):

            if params['batch_size'] < data_train_tensor.shape[1]:
                offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
            else:
                offset = 0
            batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]
     
            feed_dict_train = {x: batch_data_train}
            feed_dict_train_loss = {x: batch_data_train}
            feed_dict_val = {x: data_val_tensor}

            if (not params['been5min']) and params['auto_first']:
                sess.run(optimizer_autoencoder, feed_dict=feed_dict_train)
            else:
                sess.run(optimizer, feed_dict=feed_dict_train)

            if step % 20 == 0:
                # saves time to bunch operations with one run command (per feed_dict)
                train_errors_dict = sess.run(losses, feed_dict=feed_dict_train_loss)

                val_dicts = []
                num_val_traj = data_val_tensor.shape[1]/(params['len_time']-params['num_shifts'])
                val_batch_size = int(num_val_traj/10)
                for batch_num in xrange(10):
                    batch_data_val = data_val_tensor[:, batch_num*val_batch_size:(batch_num+1)*val_batch_size, :]
                    feed_dict_val = {x: batch_data_val}
                    batch_val_errors_dict = sess.run(losses, feed_dict=feed_dict_val)
                    val_dicts.append(batch_val_errors_dict)

                val_errors_dict = {}
                for key in val_dicts[0].keys():
                    val_errors_dict[key] = sum(d[key] for d in val_dicts) / len(val_dicts)
                
                val_error = val_errors_dict['loss']

                if val_error < (best_error - best_error * (10 ** (-5))):
                    best_error = val_error.copy()
                    saver.save(sess, params['model_path'])
                    reg_train_err = train_errors_dict['regularized_loss']
                    reg_val_err = val_errors_dict['regularized_loss']
                    print("New best val error %f (with reg. train err %f and reg. val err %f)" % (
                        best_error, reg_train_err, reg_val_err))

                train_val_error[count, 0] = train_errors_dict['loss']
                train_val_error[count, 1] = val_error
                train_val_error[count, 2] = train_errors_dict['regularized_loss']
                train_val_error[count, 3] = val_errors_dict['regularized_loss']
                train_val_error[count, 4] = train_errors_dict['loss1']
                train_val_error[count, 5] = val_errors_dict['loss1']
                train_val_error[count, 6] = train_errors_dict['loss2']
                train_val_error[count, 7] = val_errors_dict['loss2']
                train_val_error[count, 8] = train_errors_dict['loss3']
                train_val_error[count, 9] = val_errors_dict['loss3']
                train_val_error[count, 10] = train_errors_dict['loss4']
                train_val_error[count, 11] = val_errors_dict['loss4']
                train_val_error[count, 12] = train_errors_dict['loss5']
                train_val_error[count, 13] = val_errors_dict['loss5']
                train_val_error[count, 14] = train_errors_dict['reg_loss']
                train_val_error[count, 15] = val_errors_dict['reg_loss']
                if np.isnan(train_val_error[count, 3]):
                    params['stop_condition'] = 'Regularized validation loss is nan'
                    print('Regularized validation loss is nan')
                    finished = 1
                    break

                if step % 200 == 0:
                    train_val_error_trunc = train_val_error[range(count), :]
                    np.savetxt(csv_path, train_val_error_trunc, delimiter=',')
                finished, save_now = helperfns.check_progress(start, best_error, params)
                if save_now:
                    train_val_error_trunc = train_val_error[range(count), :]
                    helperfns.save_files(sess, saver, csv_path, train_val_error_trunc, params)
                if finished:
                    break
                count = count + 1

            if step > params['num_steps_per_file_pass']:
                params['stop_condition'] = 'reached num_steps_per_file_pass'
                break

    # SAVE RESULTS
    train_val_error = train_val_error[range(count), :]
    print(train_val_error)
    params['time_exp'] = time.time() - start
    saver.restore(sess, params['model_path'])
    helperfns.save_files(sess, saver, csv_path, train_val_error, params)
    
    return best_error


def main_exp(params):
    helperfns.set_defaults(params)

    if not os.path.exists(params['folder_name']):
        os.makedirs(params['folder_name'])

    # data is num_steps x num_examples x n
    data_val = np.load(('./data/%s_val_x.npy' % (params['data_name'])))

    best_error = try_net(data_val, params)
    tf.reset_default_graph()
    return best_error
