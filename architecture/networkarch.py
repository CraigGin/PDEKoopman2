import numpy as np
import tensorflow as tf
from tensorflow import keras

from DenseResBlock import DenseResBlock
from rel_mse import rel_mse

class networkarch(keras.Model):
	def __init__(self,
				 n_inputs=128,
				 n_latent=21,
				 len_time = 51,
				 num_shifts=50,
				 num_shifts_middle=50,
				 outer_encoder=DenseResBlock(),
				 outer_decoder=DenseResBlock(),
				 inner_config=dict(),
				 inner_loss_weights=[1, 1],
				 L_diag = False,
				 train_autoencoder_only=False,
				 **kwargs):
		super().__init__(**kwargs)

		self.n_inputs = n_inputs
		self.n_latent = n_latent
		self.len_time = len_time
		self.num_shifts = num_shifts
		self.num_shifts_middle = num_shifts_middle
		self.outer_encoder = outer_encoder
		self.outer_decoder = outer_decoder
		self.inner_encoder = keras.layers.Dense(n_latent,
												name='inner_encoder',
												activation=None,
												use_bias=False,
												kernel_initializer=identity_init,
												**inner_config)
		self.L = tf.Variable(tf.eye(n_latent), trainable=True)
		self.inner_decoder = keras.layers.Dense(n_inputs, 
												name='inner_decoder',
												activation=None,
												use_bias=False,
												kernel_initializer=identity_init,
												**inner_config)

		self.rel_mse = rel_mse(name='relative_mse')
		self.inner_loss_weights = inner_loss_weights
		self.L_diag = L_diag
		self.train_autoencoder_only = train_autoencoder_only

	def call(self, inputs):

		len_pred = self.len_time-self.num_shifts
		len_lin = self.len_time-self.num_shifts_middle

		# Create arrays for prediction and linearity
		pred_inputs = inputs[:,:len_pred,:]
		lin_inputs = inputs[:,:len_lin,:]
		# "Exact" solutions for linearity loss
		#lin_exact = stack_predictions(lin_inputs, self.num_shifts_middle)
		
		#  Reshape inputs as 2D arrays
		auto_inputs, pred_inputs, lin_inputs = reshape_inputs((inputs, pred_inputs, lin_inputs))
		
		# Autoencoder
		partially_encoded = self.outer_encoder(auto_inputs)
		fully_encoded = self.inner_encoder(partially_encoded)
		partially_decoded = self.inner_decoder(fully_encoded)
		autoencoder_output = self.outer_decoder(partially_decoded)
		# Outer Autoencoder
		outer_auto_output = self.outer_decoder(partially_encoded)
		# Inner Autoencoder Loss
		self.add_loss(self.inner_loss_weights[0]*self.rel_mse(partially_encoded, partially_decoded))
		# Make a tensor of all zeros for predictions
		predictions = tf.zeros([inputs.shape[0],
								len_pred*self.num_shifts,
								self.n_inputs])
		# If training autoencoder only, output results
		if self.train_autoencoder_only:
			return autoencoder_output, outer_auto_output, predictions
		# Set dynamics matrix L
		if self.L_diag:
			Lmat = tf.linalg.diag(tf.linalg.diag_part(self.L))
		else:
			Lmat = self.L
		# Prediction 
		part_encoded_pred = self.outer_encoder(pred_inputs)
		current_encoded = self.inner_encoder(part_encoded_pred)
		for shift in range(self.num_shifts):
			advanced_encoded = tf.matmul(current_encoded, Lmat)
			adv_part_decoded = self.inner_decoder(advanced_encoded)
			advanced_decoded = self.outer_decoder(adv_part_decoded)
			predictions[:,shift*len_pred:(shift+1)*len_pred,:] = tf.reshape(advanced_decoded, 
																			[inputs.shape[0], -1, self.n_inputs])
			current_encoded = advanced_encoded
		# Linearity 
		lin_pred = tf.zeros([inputs.shape[0],
							 len_lin*self.num_shifts_middle,
							 self.n_latent])
		part_encoded_lin = self.outer_encoder(lin_inputs)
		current_encoded = self.inner_encoder(part_encoded_lin)
		for shift in range(self.num_shifts_middle):
			advanced_encoded = tf.matmul(current_encoded, Lmat)
			lin_pred[:,shift*len_lin:(shift+1)*len_lin,:] = tf.reshape(advanced_encoded,
																	   [inputs.shape[0], -1, self.n_latent])
			current_encoded = advanced_encoded
		
		# Add Linearity loss
		lin_exact = lin_pred
		self.add_loss(self.inner_loss_weights[1]*self.rel_mse(lin_exact, lin_pred))
		# Reshape the outputs as 3D arrays
		autoencoder_output = tf.reshape(autoencoder_output,
										auto_inputs.shape)
		outer_auto_output = tf.reshape(outer_auto_output,
									   auto_inputs.shape)
		return autoencoder_output, outer_auto_output, predictions

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

def reshape_inputs(inputs):
	input_list = []
	for data in inputs:
		input_list.append(tf.reshape(data, [-1, data.shape[-1]]))
	return tuple(input_list)

def stack_predictions(data, num_shifts):
	len_pred = data.shape[1]-num_shifts
	prediction_tensor = np.zeros([data.shape[0], len_pred*num_shifts, data.shape[-1]])
	for j in range(num_shifts):
		prediction_tensor[:,j*len_pred:(j+1)*len_pred,:] = data[:,j+1:j+1+len_pred,:]
			
	return prediction_tensor



