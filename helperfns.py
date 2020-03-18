import datetime
import pickle
import time

import numpy as np
import scipy.io
import tensorflow as tf


def stack_data(data, num_shifts, len_time):
    nd = data.ndim
    if nd > 1:
        n = data.shape[1]
    else:
        data = (np.asmatrix(data)).getT()
        n = 1
    num_traj = data.shape[0] / len_time

    new_len_time = len_time - num_shifts

    data_tensor = np.zeros([num_shifts + 1, num_traj * new_len_time, n])

    for j in np.arange(num_shifts + 1):
        for count in np.arange(num_traj):
            data_tensor_range = np.arange(count * new_len_time, new_len_time + count * new_len_time)
            data_tensor[j, data_tensor_range, :] = data[count * len_time + j: count * len_time + j + new_len_time, :]

    return data_tensor


def choose_optimizer(params, regularized_loss, trainable_var):
    if params['opt_alg'] == 'adam':
        optimizer = tf.train.AdamOptimizer(params['learning_rate']).minimize(regularized_loss, var_list=trainable_var)
    elif params['opt_alg'] == 'adadelta':
        if params['decay_rate'] > 0:
            optimizer = tf.train.AdadeltaOptimizer(params['learning_rate'], params['decay_rate']).minimize(
                regularized_loss,
                var_list=trainable_var)
        else:
            # defaults 0.001, 0.95
            optimizer = tf.train.AdadeltaOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                     var_list=trainable_var)
    elif params['opt_alg'] == 'adagrad':
        # also has initial_accumulator_value parameter
        optimizer = tf.train.AdagradOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                var_list=trainable_var)
    elif params['opt_alg'] == 'adagradDA':
        # Be careful when using AdagradDA for deep networks as it will require careful initialization of the gradient
        # accumulators for it to train.
        optimizer = tf.train.AdagradDAOptimizer(params['learning_rate'], tf.get_global_step()).minimize(
            regularized_loss,
            var_list=trainable_var)
    elif params['opt_alg'] == 'ftrl':
        # lots of hyperparameters: learning_rate_power, initial_accumulator_value,
        # l1_regularization_strength, l2_regularization_strength
        optimizer = tf.train.FtrlOptimizer(params['learning_rate']).minimize(regularized_loss, var_list=trainable_var)
    elif params['opt_alg'] == 'proximalGD':
        # can have built-in reg.
        optimizer = tf.train.ProximalGradientDescentOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                                var_list=trainable_var)
    elif params['opt_alg'] == 'proximalAdagrad':
        # initial_accumulator_value, reg.
        optimizer = tf.train.ProximalAdagradOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                        var_list=trainable_var)
    elif params['opt_alg'] == 'RMS':
        # momentum, epsilon, centered (False/True)
        if params['decay_rate'] > 0:
            optimizer = tf.train.RMSPropOptimizer(params['learning_rate'], params['decay_rate']).minimize(
                regularized_loss,
                var_list=trainable_var)
        else:
            # default decay_rate 0.9
            optimizer = tf.train.RMSPropOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                    var_list=trainable_var)
    else:
        raise ValueError("chose invalid opt_alg %s in params dict" % params['opt_alg'])
    return optimizer


def check_progress(start, best_error, params):
    finished = 0
    save_now = 0

    current_time = time.time()

    if not params['been5min']:
        # only check 5 min progress once
        if current_time - start > 5 * 60:
            if best_error > params['min_5min']:
                print("too slowly improving in first five minutes: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first 5 min'
                finished = 1
                return finished, save_now
            else:
                print("been 5 minutes, err = %.15f < %.15f" % (best_error, params['min_5min']))
                params['been5min'] = best_error
    if not params['been20min']:
        # only check 20 min progress once
        if current_time - start > 20 * 60:
            if best_error > params['min_20min']:
                print("too slowly improving in first 20 minutes: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first 20 min'
                finished = 1
                return finished, save_now
            else:
                print("been 20 minutes, err = %.15f < %.15f" % (best_error, params['min_20min']))
                params['been20min'] = best_error
    if not params['been40min']:
        # only check 40 min progress once
        if current_time - start > 40 * 60:
            if best_error > params['min_40min']:
                print("too slowly improving in first 40 minutes: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first 40 min'
                finished = 1
                return finished, save_now
            else:
                print("been 40 minutes, err = %.15f < %.15f" % (best_error, params['min_40min']))
                params['been40min'] = best_error
    if not params['been1hr']:
        # only check 1 hr progress once
        if current_time - start > 60 * 60:
            if best_error > params['min_1hr']:
                print("too slowly improving in first hour: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first hour'
                finished = 1
                return finished, save_now
            else:
                print("been 1 hour, err = %.15f < %.15f" % (best_error, params['min_1hr']))
                save_now = 1
                params['been1hr'] = best_error
    if not params['been2hr']:
        # only check 2 hr progress once
        if current_time - start > 2 * 60 * 60:
            if best_error > params['min_2hr']:
                print("too slowly improving in first two hours: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first two hours'
                finished = 1
                return finished, save_now
            else:
                print("been 2 hours, err = %.15f < %.15f" % (best_error, params['min_2hr']))
                save_now = 1
                params['been2hr'] = best_error
    if not params['been3hr']:
        # only check 3 hr progress once
        if current_time - start > 3 * 60 * 60:
            if best_error > params['min_3hr']:
                print("too slowly improving in first three hours: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first three hours'
                finished = 1
                return finished, save_now
            else:
                print("been 3 hours, err = %.15f < %.15f" % (best_error, params['min_3hr']))
                save_now = 1
                params['been3hr'] = best_error
    if not params['been4hr']:
        # only check 4 hr progress once
        if current_time - start > 4 * 60 * 60:
            if best_error > params['min_4hr']:
                print("too slowly improving in first four hours: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first four hours'
                finished = 1
                return finished, save_now
            else:
                print("been 4 hours, err = %.15f < %.15f" % (best_error, params['min_4hr']))
                save_now = 1
                params['been4hr'] = best_error

    if not params['beenHalf']:
        # only check halfway progress once
        if current_time - start > params['max_time'] / 2:
            if best_error > params['min_halfway']:
                print("too slowly improving 1/2 of way in: val err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving halfway in'
                finished = 1
                return finished, save_now
            else:
                print("Halfway through time, err = %.15f < %.15f" % (best_error, params['min_halfway']))
                params['beenHalf'] = best_error

    if current_time - start > params['max_time']:
        params['stop_condition'] = 'past max time'
        finished = 1
        return finished, save_now

    return finished, save_now

def save_files(sess, saver, csv_path, train_val_error, params):
    np.savetxt(csv_path, train_val_error, delimiter=',')

    params['minTrain'] = np.nanmin(train_val_error[:, 0])
    params['minTest'] = np.nanmin(train_val_error[:, 1])
    params['minRegTrain'] = np.nanmin(train_val_error[:, 2])
    params['minRegTest'] = np.nanmin(train_val_error[:, 3])
    print "min train: %.12f, min val: %.12f, min reg. train: %.12f, min reg. val: %.12f" % (
        params['minTrain'], params['minTest'], params['minRegTrain'], params['minRegTest'])
    save_params(params)


def save_params(params):
    with open(params['model_path'].replace('ckpt', 'pkl'), 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
    mat_filename = params['model_path'].replace('ckpt', 'mat')
    scipy.io.savemat(mat_filename, params)


def set_defaults(params):
    if 'add_identity' not in params:
        params['add_identity'] = 0
    if 'diag_L' not in params:
        # default is that L is not forced to be diagonal
        params['diag_L'] = 0
    if 'auto_first' not in params:
        params['auto_first'] = 0
    if 'num_evals' not in params:
        raise KeyError("Error, must give number of evals: num_evals")
    if 'relative_loss' not in params:
        params['relative_loss'] = 0
    if 'num_passes_per_file' not in params:
        params['num_passes_per_file'] = 1000
    if 'num_steps_per_batch' not in params:
        params['num_steps_per_batch'] = 1
    if 'num_steps_per_file_pass' not in params:
        params['num_steps_per_file_pass'] = 1000000
    if 'data_name' not in params:
        raise KeyError("Error: must give data_name as input to main")
    if 'exp_suffix' not in params:
        params['exp_suffix'] = '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    if 'widths' not in params:
        raise KeyError("Error, must give widths as input to main")
    if 'learning_rate' not in params:
        params['learning_rate'] = .003
    if 'd' not in params:
        params['d'] = len(params['widths'])
    params['depth'] = (params['d'] - 4) / 2  # i.e. 10 or 12 -> 3 or 4
    params['num_encoder_weights'] = params['depth'] + 1
    params['num_decoder_weights'] = params['depth'] + 1
    if 'act_type' not in params:
        params['act_type'] = 'relu'
    if 'linear_encoder_layers' not in params:
        # default is that only last layer is linear
        params['linear_encoder_layers'] = [params['num_encoder_weights'] - 1, ]
        print(params['linear_encoder_layers'])
    if 'linear_decoder_layers' not in params:
        # default is that only last layer is linear
        params['linear_decoder_layers'] = [params['num_decoder_weights'] - 1, ]
        print(params['linear_decoder_layers'])
    if 'folder_name' not in params:
        params['folder_name'] = 'results'
    if 'L1_lam' not in params:
        params['L1_lam'] = .00001
    if 'Linf_lam' not in params:
        params['Linf_lam'] = 0.0
    if 'dist_weights' not in params:
        params['dist_weights'] = 'tn'
    if 'dist_biases' not in params:
        params['dist_biases'] = 0
    if 'scale' not in params:
        params['scale'] = 0.1
    if 'opt_alg' not in params:
        params['opt_alg'] = 'adam'
    if 'decay_rate' not in params:
        params['decay_rate'] = 0
    if 'batch_size' not in params:
        params['batch_size'] = 0
    if 'data_train_len' not in params:
        # default is that there's one training set
        params['data_train_len'] = 1
    if 'val_len_time' not in params:
        if 'len_time' in params:
            # if have more general len_time copy that over to len_time for validation set
            params['val_len_time'] = params['len_time']
        else:
            raise KeyError("Error, must give val_len_time or len_time as input to main")
    if 'train_len_time' not in params:
        if 'len_time' in params:
            # if have more general len_time copy that over to len_time for training set
            params['train_len_time'] = params['len_time']
        else:
            raise KeyError("Error, must give train_len_time or len_time as input to main")
    if isinstance(params['train_len_time'], int):
        params['train_len_time'] = [params['train_len_time']] * params['data_train_len']
    if 'max_time' not in params:
        params['max_time'] = 0
    if 'L2_lam' not in params:
        params['L2_lam'] = 0.0
    if 'dropout_rate' not in params:
        params['dropout_rate'] = 1.0
    if 'model_path' not in params:
        exp_name = params['data_name'] + params['exp_suffix']
        params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], exp_name)

    if 'min_5min' not in params:
        params['min_5min'] = 10 ** (-2)
    if 'min_20min' not in params:
        params['min_20min'] = 10 ** (-3)
    if 'min_40min' not in params:
        params['min_40min'] = 10 ** (-4)
    if 'min_1hr' not in params:
        params['min_1hr'] = 10 ** (-5)
    if 'min_2hr' not in params:
        params['min_2hr'] = 10 ** (-5.25)
    if 'min_3hr' not in params:
        params['min_3hr'] = 10 ** (-5.5)
    if 'min_4hr' not in params:
        params['min_4hr'] = 10 ** (-5.75)
    if 'min_halfway' not in params:
        params['min_halfway'] = 10 ** (-4)

    if 'autoencoder_only' in params:
        if params['autoencoder_only']:
            params['prediction_loss_lam'] = 0
            params['linearity_loss_lam'] = 0
    else:
        params['autoencoder_only'] = 0

    if 'inner_autoencoder_loss_lam' not in params:
        params['inner_autoencoder_loss_lam'] = 0

    if 'outer_autoencoder_loss_lam' not in params:
        params['outer_autoencoder_loss_lam'] = 0

    if 'recon_lam' in params:
        # instead of recon_lam, now have separated autoencoder_loss_lam and prediction_loss_lam
        if 'autoencoder_loss_lam' in params:
            if params['recon_lam'] != params['autoencoder_loss_lam']:
                raise KeyError(
                    "Error, replaced recon_lam with separated autoencoder_loss_lam and prediction_loss_lam, so "
                    "shouldn't have both recon_lam and autoencoder_loss_lam as keys")
            else:
                # included both, but fine because same value
                print("Note: replaced recon_lam with separated autoencoder_loss_lam and prediction_loss_lam")
        else:
            params['autoencoder_loss_lam'] = params['recon_lam']
            print("Note: replaced recon_lam with separated autoencoder_loss_lam and prediction_loss_lam")

        if 'prediction_loss_lam' in params:
            if params['recon_lam'] != params['prediction_loss_lam']:
                raise KeyError(
                    "Error, replaced recon_lam with separated autoencoder_loss_lam and prediction_loss_lam, so "
                    "shouldn't have both recon_lam and prediction_loss_lam as keys")
            else:
                # included both, but fine because same value
                print("Note: replaced recon_lam with separated autoencoder_loss_lam and prediction_loss_lam")
        else:
            params['prediction_loss_lam'] = params['recon_lam']
            print("Note: replaced recon_lam with separated autoencoder_loss_lam and prediction_loss_lam")

    if 'mid_shift_lam' in params:
        # renamed mid_shift_lam to linearity_loss_lam
        if 'linearity_loss_lam' in params:
            if params['linearity_loss_lam'] != params['mid_shift_lam']:
                raise KeyError("Error, renamed mid_shift_lam to linearity_loss_lam, so shouldn't have both as keys")
            else:
                # included both, but fine because same value
                print("Note: renamed mid_shift_lam to linearity_loss_lam")
        else:
            # called it mid_shift_lam instead of linearity_loss_lam
            params['linearity_loss_lam'] = params['mid_shift_lam']
            print("Note: renamed mid_shift_lam to linearity_loss_lam")
    if 'linearity_loss_lam' not in params:
        params['linearity_loss_lam'] = 1.0

    if 'shifts' not in params:
        if 'num_shifts' in params:
            params['shifts'] = np.arange(params['num_shifts']) + 1
        else:
            # shifts AND num_shifts not in params: fine only if prediction_loss_lam = 0
            if params['prediction_loss_lam']:
                raise KeyError('Have nonzero prediction_loss_lam but not shifts or num_shifts')
            else:
                params['shifts'] = []
                params['num_shifts'] = 0

    if 'shifts_middle' not in params:
        if 'num_shifts_middle' in params:
            params['shifts_middle'] = np.arange(params['num_shifts_middle']) + 1
        else:
            # shifts_middle AND num_shifts_middle not in params: fine only if linearity_loss_lam = 0
            if params['linearity_loss_lam']:
                raise KeyError('Have nonzero linearity_loss_lam but not shifts_middle or num_shifts_middle')
            else:
                params['shifts_middle'] = []
                params['num_shifts_middle'] = 0

    if 'num_shifts' not in params:
        params['num_shifts'] = len(params['shifts'])
    if 'num_shifts_middle' not in params:
        params['num_shifts_middle'] = len(params['shifts_middle'])

    if 'autoencoder_loss_lam' not in params:
        params['autoencoder_loss_lam'] = 1.0
    if 'prediction_loss_lam' not in params:
        params['prediction_loss_lam'] = 1.0

    params['been5min'] = 0
    params['been20min'] = 0
    params['been40min'] = 0
    params['been1hr'] = 0
    params['been2hr'] = 0
    params['been3hr'] = 0
    params['been4hr'] = 0
    params['beenHalf'] = 0

    if isinstance(params['dist_weights'], basestring):
        params['dist_weights'] = [params['dist_weights']] * (len(params['widths']) - 1)
    if isinstance(params['dist_biases'], int):
        params['dist_biases'] = [params['dist_biases']] * (len(params['widths']) - 1)

    if params['autoencoder_only']:
        # then shouldn't have any shifts or shifts_middle
        if params['num_shifts']:
            raise ValueError('doing autoencoder only, so should have num_shifts = 0')
        if params['num_shifts_middle']:
            raise ValueError('doing autoencoder only, so should have num_shifts_middle = 0')
    if 'mu' not in params:
        params['mu'] = 1.0
    if 'fix_middle' not in params:
        params['fix_middle'] = 0
    if 'seed_middle' not in params:
        params['seed_middle'] = 0
        params['fix_middle'] = 0
    if 'restore' not in params:
        params['restore'] = 0
    return params


def num_shifts_in_stack(params):
    max_shifts_to_stack = 0
    if params['num_shifts']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if params['num_shifts_middle']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))

    return max_shifts_to_stack

def identity_seed(n_rows,n_cols):
    if n_rows >= n_cols:
        A = np.zeros((n_rows,n_cols), dtype=np.float32)
        for col in xrange(n_cols):
            for row in xrange(col,col+n_rows-n_cols+1):
                A[row,col] = 1.0/(n_rows-n_cols+1) 
    else:
        A = np.zeros((n_rows,n_cols), dtype=np.float32)
        for row in xrange(n_rows):
            for col in xrange(row,row+n_cols-n_rows+1):
                A[row,col] = 1.0/(n_cols-n_rows+1) 

    return A    

def DFT_matrix(dim):
    DFT = dft(dim) 
    rDFT = np.real(DFT)
    iDFT = np.imag(DFT)
    combinedDFT = rDFT[0,:]
    for i in xrange(1,n_inputs/2):
        combinedDFT = np.vstack((combinedDFT, rDFT[i,:]))
        combinedDFT = np.vstack((combinedDFT, iDFT[i,:]))
    combinedDFT = np.vstack((combinedDFT, rDFT[n_inputs/2,:])) 

    return combinedDFT

def reduced_DFT(large_dim,small_dim):
    combinedDFT = DFT_matrix(large_dim)
    Reduce = np.hstack((np.eye(small_dim),np.zeros((small_dim,large_dim-small_dim))))
    Reduced_DFT = Reduce.dot(combinedDFT)
    Reduced_DFT = np.float32(Reduced_DFT.T)

    return Reduced_DFT

def expand_IDFT(small_dim,large_dim):
    combinedDFT = DFT_matrix(large_dim)
    Expand = np.vstack((np.eye(small_dim),np.zeros((large_dim-small_dim,small_dim))))
    inv_DFT = np.linalg.inv(combinedDFT)
    Expand_IDFT = inv_DFT.dot(Expand)
    Expand_IDFT = np.float32(Expand_IDFT.T)

    return Expand_IDFT

def freq_vector(dim):
    max_freq = np.divide(dim,2)
    kv = np.empty((dim,))
    if dim % 2 == 0:
        kv[::2] = np.array(range(max_freq))
    else:
        kv[::2] = np.array(range(max_freq+1))
    kv[1::2] = np.array(range(1,max_freq+1))

    return kv


