# PDEKoopman2 - Using neural networks to learn linearizing transformations for PDEs

This code implements the method found in the paper ["Deep Learning Models for Global Coordinate Transformations that Linearise PDEs"](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/deep-learning-models-for-global-coordinate-transformations-that-linearise-pdes/4C3252EA5D681D07D933AD31EE539192) by Craig Gin, Bethany Lusch, Steven L. Brunton, and J. Nathan Kutz. 

**Note:** The code used to produce the results of the paper can be found at https://github.com/CraigGin/PDEKoopman. If you simply wish to verify the results of the paper, you should use that code which was written for Python 2 and Tensorflow 1. If, however, you wish to implement the method for your own problem, you should use this code. This repository contains signifantly cleaner and simpler code that is written for current versions of Python and Tensorflow. 

## Instructions for running the code

### 1. Clone the repository

```
git clone https://github.com/CraigGin/PDEKoopman2.git
```

### 2. Add or create data

You need to add or create data in the `data` directory. All data should be in the form of 3D NumPy arrays stored in .npy files. The code assumes that the training and validation data have already been split. The training data can consist of a single file or multiple files. If a single file, it should be named

`dataname_train1_x.npy`

where `dataname` can be chosen to describe your data set. For large data sets that need to be split over multiple files, subsequent files should use the same naming convention:

`dataname_train2_x.npy`, `dataname_train3_x.npy`, ...

Validation data should be in a single file named

`dataname_val_x.npy`

The NumPy arrays should consist of trajectories from your dynamical system (i.e. solutions to a PDE). The shape of each array should be

(number of trajectories, length of trajectory, state space dimension)

which can alternatively be described as 

(number of initial conditions, length of time discretization, number of points in spatial discretization). 

Included with the repository are scripts that create the data from the paper. The script `Burgers_Eqn_data.py` creates 20 training data files and a validation data file with solutions to Burgers' equation. The script `Burgers_Eqn_testdata.py` creates 5 test data files (for the five types of initial conditions described in the paper). Similarly, `KS_Eqn_data.py` creates training and validation data for the Kuramoto-Sivashinsky equation and `KS_Eqn_testdata.py` creates test data. The randomization for the initial conditions uses the Python package [pyDOE](https://pythonhosted.org/pyDOE/index.html) so you must install that package to generate the data. Note that the data generation scripts take a long time to run. A faster alternative is to download the data files directly from [PDEKoopman Data](https://drive.google.com/drive/u/0/folders/1T09uKjvNf-LxjOCQVJyOpoWOntvYghV-). 

### 3. Run the experiment 

In the `experiments` directory, create an experiment file. Two examples have been provided. `Sample_Dense_Expt.py` trains an autoencoder with fully connected layers in the encoder and decoder as was done for Burgers' equation in the paper (see Figure 10 for the architecture). `Sample_Conv_Expt.py` trains a convolutional autoencoder as was done for the KS equation in the paper (see Figure 16 for the architecture). Each of these experiment files trains 20 networks with randomly chosen learning rates and initializations for 18 epochs (the first three of which use only the three autoencoder losses). Then the best network (based on total validation loss) is chosen and trained for several hundred epochs (500 and 300, respectively). During the training process, the networks are checkpointed and saved in the `model_weights` directory, which is created by the experiment script. After training is completed, the final network weights and information about the training process are saved in the `results` directory.

The following is a description of the parameters that must be specified in the experiment file:

* expt_name - name for your experiment (sets the name of the subdirectories where results are stored)
* data_file_prefix - the relative path and `dataname` for your data files. For example, if your data files are named `Burgers_Eqn_train1_x.npy`, ..., `Burgers_Eqn_train20_x.npy`, `Burgers_Eqn_val_x.npy`, then you should use `..data/Burgers_Eqn`.
* training_options - a dictionary with keyword arguments for training options. The dictionary contains the following, some of which are defined as variables in the sample scripts prior to creating the training_options disctionary:
  * aec_only_epochs - the number of epochs to train each initial model with only the autoencoder losses
  * init_full_epochs - the number of epochs to train each initial model with all losses
  * best_model_epochs - the number of epochs to train the final model after the best initial model is chosen and loaded
  * num_init_models - the number of initial models to train, the best of which is trained to convergence
  * loss_fn - the loss function. In the paper, we use a relative mean-squared error. This is implemented by subclassing the Keras Loss class (see `architecture/RelMSE.py`). You can create your own loss function or use one of the built-in Keras losses.
  * optimizer - a Keras optimizer
  * optimizer_opts - a dictionary of keyword arguments to feed to the Keras optimizer
  * batch_size - the batch size used for training
  * data_train_len - the number of training data files
  * loss_weights - a list of the relative weights given to each loss function (in the order they are listed in the paper).
* network_config - a dictionary with keyword arguments for the neural network configuration. The dictionary contains the following, some of which are defined as variables in the sample scripts prior to creating the network_config disctionary: 
  * n_inputs - the number of inputs to the network. This is automatically calculated in the sample scripts by checking the size of the data arrays.
  * n_latent - the dimension of the latent space (i.e. the rank of the reduced order model)
  * len_time - the length of the trajectories in the data. This is automatically calculated in the sample scripts by checking the size of the data arrays.
  * num_shifts - the number of time steps included in the prediction loss (this is not described in the paper - see #num_shifts-and-num_shifts_middle below for details). For the paper, this was always len_time - 1.
  * num_shifts_middle - the number of time steps included in the linearity loss (this is not described in the paper - see #num_shifts-and-num_shifts_middle below for details). For the paper, this was always len_time - 1.
  * outer_encoder - a Keras model for the outer encoder of the network. For the sample experiments, we subclassed the Keras Layer class (see `architecture/DenseResBlock.py` and `architecture/ConvResBlock.py`). If you want to use the same network architecture as the paper, you can use these subclassed models and set the parameters in the sample experiment (see details below in #Using-the-included-architectures-for-the-outer-encoder/decoder ). If you want to change the network architecture, you can create your own Keras model using the Sequential API, the Functional API, or by subclassing.
  * outer_decoder - a Keras model for the outer decoder of the network. See outer_encoder above for more details.
  * inner_config - a dictionary with keyword arguments for the inner encoder and inner decoder layer. This can include any arguments accepted by keras.layers.Dense except for the following which are hard-coded: units, name, activation, use_bias, and kernel_initializer.
  * L_diag - a boolean that sets whether the dynamics matrix L is contrained to be diagonal (True) or allowed to be any square matrix (False)
* custom_objs - a dictionary that sets custom objects for loading the models. In our case, this only needs to be the custom loss function RelMSE. If using a built-in Keras loss, this dictionary can be empty.

#### num_shifts and num_shifts_middle

Add a description here

#### Using the included architectures for the outer encoder/decoder

Add a description here

### 4. Process results

Process results with notebook.

## Questions/Comments/Bugs

If you have any questions about the code or find any bugs, feel free to either email me directly at crgin@nscu.edu or create an issue. Additionally, if you are using our code for a project that you are working on, we would love to hear about it. 

## Contributors

This code was written by:

* Craig Gin - https://github.com/CraigGin
* Bethany Lusch - https://github.com/BethanyL
* Dan Shea - https://github.com/sheadan

## Dependencies

This code requires Tensorflow version 2.2.0 or newer. Additionally, the data generation scripts use the package [pyDOE](https://pythonhosted.org/pyDOE/index.html).
