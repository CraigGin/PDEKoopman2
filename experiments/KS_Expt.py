#!/usr/bin/python3
import random as r
import sys

from tensorflow import keras
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.activations import relu

from utils import run_experiment, getdatasize

# Add the architecture path for the DenseResBlock and rel_mse
sys.path.append("../architecture/")
from DenseResBlock import DenseResBlock
from rel_mse import rel_mse

# Example Experiment Script:
expt_name = 'KS_Expt'
data_file_prefix = '../data/KS_Eqn'

# Set size of latent space, and retrieve the 'full' size of the data
n_latent = 21
_, len_time, n_inputs = getdatasize(data_file_prefix)

# Set other parameters
data_train_len = 20  # Number of training data files
L_diag = False  # Whether the dynamics matrix is forced to be diagonal
num_shifts = 50
num_shifts_middle = 50
loss_weights = [1, 1, 1, 1, 1]

# Set up encoder and decoder configuration dict(s)
activation = relu
initializer = keras.initializers.VarianceScaling()
regularizer = l1_l2(0, 1e-8)

hidden_config = {'activation': activation,
                 'kernel_initializer': initializer,
                 'kernel_regularizer': regularizer}

output_config = {'activation': None,
                 'kernel_initializer': initializer,
                 'kernel_regularizer': regularizer}

outer_config = {'n_inputs': n_inputs,
                'num_hidden': 4,
                'hidden_config': hidden_config,
                'output_config': output_config}

inner_config = {'kernel_regularizer': regularizer}

# Network configuration (this is how the AbstractArchitecture will be created)
network_config = {'n_inputs': n_inputs,
                  'n_latent': n_latent,
                  'len_time': len_time,
                  'num_shifts': num_shifts,
                  'num_shifts_middle': num_shifts_middle,
                  'outer_encoder': DenseResBlock(**outer_config),
                  'outer_decoder': DenseResBlock(**outer_config),
                  'inner_config': inner_config,
                  'L_diag': L_diag}

# Aggregate all the training options in one dictionary
training_options = {'aec_only_epochs': 3,
                    'init_full_epochs': 15,
                    'best_model_epochs': 300,
                    'num_init_models': 20,
                    'loss_fn': rel_mse(),
                    'optimizer': keras.optimizers.Adam,
                    'optimizer_opts': {},
                    'batch_size': 32,
                    'data_train_len': data_train_len,
                    'loss_weights': loss_weights}

#
# Launch the Experiment
#

# Get a random number generator seed
random_seed = r.randint(0, 10**(10))

# Set the custom objects used in the model (for loading purposes)
custom_objs = {"rel_mse": rel_mse}

# And run the experiment!
run_experiment(random_seed=random_seed,
               expt_name=expt_name,
               data_file_prefix=data_file_prefix,
               training_options=training_options,
               network_config=network_config,
               custom_objects=custom_objs)
