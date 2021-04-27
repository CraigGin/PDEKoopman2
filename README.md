# PDEKoopman2 - Using neural networks to learn linearizing transformations for PDEs

This code implements the method found in the paper ["Deep Learning Models for Global Coordinate Transformations that Linearise PDEs"](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/deep-learning-models-for-global-coordinate-transformations-that-linearise-pdes/4C3252EA5D681D07D933AD31EE539192) by Craig Gin, Bethany Lusch, Steven L. Brunton, and J. Nathan Kutz. 

**Note:** The code used to produce the results of the paper can be found at https://github.com/CraigGin/PDEKoopman. If you simply wish to verify the results of the paper, you should use that code which was written for Python 2 and Tensorflow 1. If, however, you wish to implement the method for your own problem, you should use this code. This repository contains significantly cleaner and simpler code that is written for current versions of Python and Tensorflow. 

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
* data_file_prefix - the relative path and `dataname` for your data files. For example, if your data files are named `Burgers_Eqn_train1_x.npy`, ..., `Burgers_Eqn_train20_x.npy`, `Burgers_Eqn_val_x.npy`, then you should use `../data/Burgers_Eqn`.
* training_options - a dictionary with keyword arguments for training options. The dictionary contains the following, some of which are defined as variables in the sample scripts prior to creating the training_options dictionary:
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
* network_config - a dictionary with keyword arguments for the neural network configuration. The dictionary contains the following, some of which are defined as variables in the sample scripts prior to creating the network_config dictionary: 
  * n_inputs - the number of inputs to the network. This is automatically calculated in the sample scripts by checking the size of the data arrays.
  * n_latent - the dimension of the latent space (i.e. the rank of the reduced order model)
  * len_time - the length of the trajectories in the data. This is automatically calculated in the sample scripts by checking the size of the data arrays.
  * num_shifts - the number of time steps included in the prediction loss (this is not described in the paper - see [num_shifts and num_shifts_middle](#num_shifts-and-num_shifts_middle) below for details). For the paper, this was always len_time - 1.
  * num_shifts_middle - the number of time steps included in the linearity loss (this is not described in the paper - see [num_shifts and num_shifts_middle](#num_shifts-and-num_shifts_middle) below for details). For the paper, this was always len_time - 1.
  * outer_encoder - a Keras model for the outer encoder of the network. For the sample experiments, we subclassed the Keras Layer class (see `architecture/DenseResBlock.py` and `architecture/ConvResBlock.py`). If you want to use the same network architecture as the paper, you can use these subclassed models and set the parameters in the sample experiment (see details below in [Using the included architectures for the outer encoder/decoder](#Using-the-included-architectures-for-the-outer-encoderdecoder)). If you want to change the network architecture, you can create your own Keras model using the Sequential API, the Functional API, or by subclassing.
  * outer_decoder - a Keras model for the outer decoder of the network. See outer_encoder above for more details.
  * inner_config - a dictionary with keyword arguments for the inner encoder and inner decoder layer. This can include any arguments accepted by keras.layers.Dense except for the following which are hard-coded: units, name, activation, use_bias, and kernel_initializer.
  * L_diag - a boolean that sets whether the dynamics matrix L is contrained to be diagonal (True) or allowed to be any square matrix (False)
* custom_objs - a dictionary that sets custom objects for loading the models. In our case, this only needs to be the custom loss function RelMSE. If using a built-in Keras loss, this dictionary can be empty.

#### num_shifts and num_shifts_middle

The parameter num_shifts controls the number of time steps that the network predicts forward in time when calculating the prediction loss. In the paper, each trajectory contains 51 time steps (the initial condition and M = 50 steps forward in time). We used num_shifts = 50. Because the data contains 51 time steps, only the initial condition has labels for the prediction 50 steps forward in time. Therefore, only the initial condition was used for prediction and we evaluated the prediction accuracy for 1 to 50 steps forward in time. This is given by the loss function (see Equation 2.18):

![L4](https://latex.codecogs.com/svg.latex?L_4%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bj%3D1%7D%5E%7BN%7D%20%5Cfrac%7B1%7D%7BM%7D%20%5Csum_%7Bp%3D1%7D%5E%7BM%7D%20%5Cfrac%7B%5Cleft%5ClVert%20%5Cmathbf%7Bu%7D%5Ej_p%20-%20%5Cvarphi_d%28%5Cmathbf%7BK%7D%5Ep%5Cvarphi_e%28%5Cmathbf%7Bu%7D%5Ej_0%29%20%5Cright%5CrVert_2%5E2%7D%7B%5Cleft%5ClVert%20%5Cmathbf%7Bu%7D%5Ej_p%20%5Cright%5CrVert_2%5E2%7D.)

where M = 50.

However, the code also gives the option to predict fewer time steps into the future. Suppose you choose num_shifts = 10. Then the first 41 time points (i.e. M-num_shifts+1) in each trajectory have labels up to 10 steps forward in time in the data. So for each trajectory, the network would predict 1 to 10 time steps into the future for the first 41 time steps. The loss function in this case is:

![numshifts10](https://latex.codecogs.com/gif.latex?L_4%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bj%3D1%7D%5E%7BN%7D%20%5Cfrac%7B1%7D%7B10%7D%20%5Csum_%7Bp%3D1%7D%5E%7B10%7D%20%5Cfrac%7B1%7D%7B41%7D%20%5Csum_%7Bi%3D0%7D%5E%7B40%7D%20%5Cfrac%7B%5Cleft%5ClVert%20%5Cmathbf%7Bu%7D%5Ej_%7Bp&plus;i%7D%20-%20%5Cvarphi_d%28%5Cmathbf%7BK%7D%5Ep%5Cvarphi_e%28%5Cmathbf%7Bu%7D%5Ej_i%29%20%5Cright%5CrVert_2%5E2%7D%7B%5Cleft%5ClVert%20%5Cmathbf%7Bu%7D%5Ej_%7Bp&plus;i%7D%20%5Cright%5CrVert_2%5E2%7D.)

More generally, if S = num_shifts, the prediction loss is

![numshifts](https://latex.codecogs.com/gif.latex?L_4%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bj%3D1%7D%5E%7BN%7D%20%5Cfrac%7B1%7D%7BS%7D%20%5Csum_%7Bp%3D1%7D%5E%7BS%7D%20%5Cfrac%7B1%7D%7BM-S&plus;1%7D%20%5Csum_%7Bi%3D0%7D%5E%7BM-S%7D%20%5Cfrac%7B%5Cleft%5ClVert%20%5Cmathbf%7Bu%7D%5Ej_%7Bp&plus;i%7D%20-%20%5Cvarphi_d%28%5Cmathbf%7BK%7D%5Ep%5Cvarphi_e%28%5Cmathbf%7Bu%7D%5Ej_i%29%20%5Cright%5CrVert_2%5E2%7D%7B%5Cleft%5ClVert%20%5Cmathbf%7Bu%7D%5Ej_%7Bp&plus;i%7D%20%5Cright%5CrVert_2%5E2%7D.)

The parameter num_shifts_middle plays a similar role but for prediction in the latent space (i.e. for the linearity loss).

The practical implications are that larger numbers for num_shifts and num_shifts_middle produce networks that can reliably predict further into the future. However, the network is more difficult to train and sometimes the network doesn't converge to anything useful. If you are getting bad results with large values for num_shifts and num_shifts_middle, you may want to consider pre-training with small values and then reloading the pre-trained network and training with larger values.

#### Using the included architectures for the outer encoder/decoder

If you would like to use a residual network with fully connected layers for the outer encoder/decoder (like we did for Burgers' equation in the paper), you can use the `DenseResBlock` class as demonstrated in `Sample_Dense_Expt.py`. The following are the parameters which can be specified for the `DenseResBlock` class:

* n_inputs - the number of inputs to the network. This is automatically calculated in the sample scripts by checking the size of the data arrays.
* num_hidden - the number of hidden layers in the dense residual block
* hidden_config - a dictionary with keyword arguments for the hidden layers. These arguments will be used by keras.layers.Dense and can include things like activation, kernel_initializer, and kernel_regularizer.
* output_config - a dictionary with keyword arguments for the output layer of the residual block. These arguments will be used by keras.layers.Dense and can include things like activation, kernel_initializer, and kernel_regularizer It is recommended that this layer be linear (i.e. no activation function).

If you would like to use a residual network with convolutional layers for the outer encoder/decoder (like we did for the KS equation in the paper), you can use the `ConvResBlock` class as demonstrated in `Sample_Conv_Expt.py`. The following are the parameters which can be specified for the `ConvResBlock` class:

* n_inputs - the number of inputs to the network. This is automatically calculated in the sample scripts by checking the size of the data arrays.
* num_filters - a list with the number of filters for each convolutional layer. The length of the list determines the number of convolutional layers.
* convlay_config - a dictionary with keyword arguments for the convolutional layers. These arguments will be used by keras.layers.Conv1D and should include things like kernel_size, strides, padding, activation, kernel_initializer, and kernel_regularizer.
* poollay_config - a dictionary with keyword arguments for the pooling layers. These arguments will be used by keras.layers.AveragePooling1D and can include things like pool_size, strides, and padding.
* dense_config - a dictionary with keyword arguments for the hidden dense layers. These arguments will be used by keras.layers.Dense and can include things like activation function, initializer, and regularizer.
* output_config - a dictionary with keyword arguments for the output layer of the residual block. These arguments will be used by keras.layers.Dense and can include things like activation function, initializer, and regularizer. It is recommended that this layer be linear (i.e. no activation function).


### 4. Process results

Included is a Jupyter notebook `process_results.ipynb` which gives an example of how you might load the model to do prediction and view some of the results saved in the `results` directory. The `results` directory currently has results created by `Sample_Dense_Expt.py` and `Sample_Conv_Expt.py` for Burgers' equation and the KS equation, respectively. You can run this notebook (assuming you have the data in the `data` directory) to see results similar to the paper.

## Questions/Comments/Bugs

If you have any questions about the code or find any bugs, feel free to either email me directly at crgin@nscu.edu or create an issue. Additionally, if you are using our code for a project that you are working on, we would love to hear about it. 

## Contributors

This code was written by:

* Craig Gin - https://github.com/CraigGin
* Bethany Lusch - https://github.com/BethanyL
* Dan Shea - https://github.com/sheadan

## Dependencies

This code requires Tensorflow version 2.2.0 or newer. Additionally, the data generation scripts use the package [pyDOE](https://pythonhosted.org/pyDOE/index.html).
