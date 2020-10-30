# PDEKoopman2 - Using neural networks to learn linearizing transformations for PDEs

This code implements the method found in the paper ["Deep Learning Models for Global Coordinate Transformations that Linearise PDEs"](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/deep-learning-models-for-global-coordinate-transformations-that-linearise-pdes/4C3252EA5D681D07D933AD31EE539192) by Craig Gin, Bethany Lusch, Steven L. Brunton, and J. Nathan Kutz. 

**Note:** The code used to produce the results of the paper can be found at https://github.com/CraigGin/PDEKoopman. If you simply wish to verify the results of the paper, you should use that code which was written for Python 2 and Tensorflow 1. If, however, you wish to implement the method for your own problem, you should use this code. This repository contains signifantly cleaner and simpler code that is written for current versions of Python and Tensorflow. 

## Instructions for running the code

### 1. Clone the repository.

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

(number of initial conditions, length of time discretization, number of point in spatial discretization). 

Included with the repository are scripts that create the data from the paper. The script `Burgers_Eqn_data.py` creates 20 training data files and a validation data file. The script `Burgers_Eqn_testdata.py` creates 5 test data files (for the five types of initial conditions described in the paper). Similarly, `KS_Eqn_data.py` creates training and validation data for the Kuramoto-Sivashinsky equation and `KS_Eqn_testdata.py` creates test data. The randomization for the initial conditions uses the Python package [pyDOE](https://pythonhosted.org/pyDOE/index.html) so you must install that package to generate the data. Note that the data generation scripts take a long time to run. So an alternative is to download the data files directly from [PDEKoopman Data](https://drive.google.com/drive/u/0/folders/1T09uKjvNf-LxjOCQVJyOpoWOntvYghV-). 

### 3. Run the experiment 

In the experiments directory, edit the desired experiment files. As an example, Burgers_Experiment_28rr.py will train 20 neural networks with randomly chosen learning rates and initializations each for 20 minutes. It will create a directory called Burgers_exp28rr and store the networks and losses. You can then run the file Burgers_Experiment28rr_restore.py to restore the network with the smallest validation loss and continue training the network until convergence.
3b. Can add network architectures in architecture folder

### 4. Process results

Process results with notebook.

## Questions/Comments/Bugs

If you have any questions about the code or find any bugs, feel free to either email me directly at crgin@nscu.edu or raise an issue

## Contributors

This code was written by:

* Craig Gin - https://github.com/CraigGin
* Bethany Lusch - https://github.com/BethanyL
* Dan Shea - https://github.com/sheadan

## Dependencies

This code requires Tensorflow version 2.2.0 or newer. Additionally, the data generation scripts use the package [pyDOE](https://pythonhosted.org/pyDOE/index.html).
