import tensorflow as tf
import numpy as np
from util import *



class Model():
	def __init__(self, args, data):
		# initialize parameters
		self.sequence_length = args.sequence_length
		self.layer_num = args.layer_num
		self.neg_samples = args.neg_samples

		self.learn_rate = args.learn_rate
		self.latent_dimension = args.latent_dimension

		self.user_size = data.user_size
		self.item_size = data.item_size


		
	def forward(self, batch_sequences, batch_items, is_training):	
		# embedding layer
		with tf.variable_scope("embedding_layer", reuse = tf.AUTO_REUSE):
			item_embeddings = tf.get_variable("item_embeddings", shape = [self.item_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			item_embeddings = tf.concat([tf.zeros(shape=[1, self.latent_dimension]), item_embeddings[1:, :]], 0)
			sequences = tf.nn.embedding_lookup(item_embeddings, batch_sequences)

		# gru layer
		gru_cell = tf.nn.rnn_cell.GRUCell(self.latent_dimension)
		gru_cells = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * self.layer_num)

		initial_state = gru_cells.zero_state(tf.shape(sequences)[0], tf.float32)
		with tf.variable_scope("dynamic_rnn", reuse = tf.AUTO_REUSE):
			outputs, _ = tf.nn.dynamic_rnn(gru_cells, sequences, initial_state = initial_state)
		output = outputs[:, -1, :]

		# fully connected layer
		with tf.variable_scope("fc_layer", reuse = tf.AUTO_REUSE):
			fc_weights = tf.get_variable(name = "fc_weights", shape = [self.item_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			pred_embs = tf.nn.embedding_lookup(fc_weights, batch_items)
			logits = tf.squeeze(tf.matmul(pred_embs, tf.expand_dims(output, -1)), -1)

		# if train, return loss and train_op; if test, return predictions
		if is_training:
			tar_logits, neg_logits = logits[:, :1], logits[:, 1:]
			tar_logits = tf.tile(tar_logits, [1, self.neg_samples])
			#loss = tf.reduce_mean(-tf.log(tf.sigmoid(tar_logits - neg_logits) + 1e-8))
			loss = tf.reduce_mean(tf.sigmoid(neg_logits - tar_logits) + tf.sigmoid(neg_logits * neg_logits))
			train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)
			return loss, train_op
		else:
			predictions = tf.nn.softmax(logits)
			return predictions

