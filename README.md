# PDEKoopman2
## Using neural networks to learn linearizing transformations for PDEs

This code implements the method found in the paper ["Deep Learning Models for Global Coordinate Transformations that Linearise PDEs"](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/deep-learning-models-for-global-coordinate-transformations-that-linearise-pdes/4C3252EA5D681D07D933AD31EE539192) by Craig Gin, Bethany Lusch, Steven L. Brunton, and J. Nathan Kutz. The code used to produce the results of the paper can be found at https://github.com/CraigGin/PDEKoopman. If you simply wish to verify the results of the paper, you should use that code which was written for Python 2 and Tensorflow 1. If, however, you wish to implement the method for your own problem, you should use this code. This repository contains signifantly cleaner and simpler code that is written for current versions of Python and Tensorflow. 

Packages required:
Python version?
Tensorflow 2.2.0 or newer
PyDOE for data generation

To run the code:

1. Clone the repository.
2. In the data directory, create or add data. npy files with shape (num_examples, num_times, state_space_dimension). Data from paper can come from included files or downloaded.
3. In the experiments directory, edit the desired experiment files. As an example, Burgers_Experiment_28rr.py will train 20 neural networks with randomly chosen learning rates and initializations each for 20 minutes. It will create a directory called Burgers_exp28rr and store the networks and losses. You can then run the file Burgers_Experiment28rr_restore.py to restore the network with the smallest validation loss and continue training the network until convergence.
3b. Can add network architectures in architecture folder
4. Process results with notebook.
