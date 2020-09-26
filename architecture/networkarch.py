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

		self.rel_mse = rel_mse(name='rel_mse')
		self.inner_loss_weights = inner_loss_weights
		self.L_diag = L_diag
		self.train_autoencoder_only = train_autoencoder_only

	def call(self, inputs):

		len_pred = self.len_time-self.num_shifts
		len_lin = self.len_time-self.num_shifts_middle

		# Create arrays for prediction and linearity
		#pred_inputs = tf.slice(inputs, begin=[0, 0, 0], size=[-1, len_pred, -1])
		pred_inputs = inputs[:,:len_pred,:]
		#lin_inputs = tf.slice(inputs, begin=[0, 0, 0], size=[-1, len_lin, -1])
		lin_inputs = inputs[:,:len_lin,:]

		# "Exact" solutions for linearity loss
		pred_exact = stack_predictions(inputs, self.num_shifts)
		lin_advanced = stack_predictions(inputs, self.num_shifts_middle)
		
		#  Reshape inputs as 2D arrays
		auto_inputs, pred_inputs, lin_inputs, lin_advanced = reshape_inputs((inputs, pred_inputs, lin_inputs, lin_advanced))
		
		# Autoencoder
		partially_encoded = self.outer_encoder(auto_inputs)
		fully_encoded = self.inner_encoder(partially_encoded)
		partially_decoded = self.inner_decoder(fully_encoded)
		autoencoder_output = self.outer_decoder(partially_decoded)

		autoencoder_output = tf.reshape(autoencoder_output,
										[-1, self.len_time, self.n_inputs])
		
		# Outer Autoencoder
		outer_auto_output = self.outer_decoder(partially_encoded)

		outer_auto_output = tf.reshape(outer_auto_output,
									   [-1, self.len_time, self.n_inputs])
		
		# Inner Autoencoder Loss
		self.add_loss(self.inner_loss_weights[0]*self.rel_mse(partially_encoded, partially_decoded))

		# If training autoencoder only, output results
		if self.train_autoencoder_only:
			predictions = 0*pred_exact
			return autoencoder_output, outer_auto_output, predictions
		
		# Set dynamics matrix L
		if self.L_diag:
			Lmat = tf.linalg.diag(tf.linalg.diag_part(self.L))
		else:
			Lmat = self.L
		
		# Prediction 
		predictions_list = []
		part_encoded_pred = self.outer_encoder(pred_inputs)
		current_encoded = self.inner_encoder(part_encoded_pred)
		for shift in range(self.num_shifts):
			advanced_encoded = tf.matmul(current_encoded, Lmat)
			adv_part_decoded = self.inner_decoder(advanced_encoded)
			advanced_decoded = self.outer_decoder(adv_part_decoded)
			predictions_list.append(tf.reshape(advanced_decoded, 
											   [-1, len_pred, self.n_inputs]))
			current_encoded = tf.identity(advanced_encoded)
		predictions = tf.concat(predictions_list, axis=1)

		# Linearity 
		linearity_list = []
		part_encoded_lin = self.outer_encoder(lin_inputs)
		current_encoded = self.inner_encoder(part_encoded_lin)
		for shift in range(self.num_shifts_middle):
			advanced_encoded = tf.matmul(current_encoded, Lmat)
			current_encoded = tf.identity(advanced_encoded)
			linearity_list.append(tf.reshape(current_encoded,
											[-1, len_lin, self.n_latent]))	
		lin_pred = tf.concat(linearity_list, axis=1)

		lin_part_encoded = self.outer_encoder(lin_advanced)
		lin_exact = self.inner_encoder(lin_part_encoded)
		lin_exact = tf.reshape(lin_exact, [-1, self.num_shifts_middle*len_lin, self.n_latent])

		# Add Linearity loss
		self.add_loss(self.inner_loss_weights[1]*self.rel_mse(lin_exact, lin_pred))
		
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
	prediction_list = []
	for j in range(num_shifts):
		prediction_list.append(data[:,j+1:j+1+len_pred,:])
	prediction_tensor = tf.concat(prediction_list, axis=1)
			
	return prediction_tensor



