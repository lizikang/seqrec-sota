import tensorflow as tf



class Model():
	def __init__(self, args, data):
		# initialize parameters
		self.sequence_length = args.sequence_length
		self.target_length = args.target_length

		self.learn_rate = args.learn_rate
		self.latent_dimension = args.latent_dimension
		self.keep_prob = args.keep_prob

		self.user_size = data.user_size
		self.item_size = data.item_size


	def forward(self, batch_users, batch_sequences, batch_items, is_training):	
		# embedding layer
		with tf.variable_scope("embedding_layer", reuse = tf.AUTO_REUSE):
			user_embeddings = tf.get_variable("user_embeddings", shape = [self.user_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			item_embeddings = tf.get_variable("item_embeddings", shape = [self.item_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			item_embeddings = tf.concat([tf.zeros(shape=[1, self.latent_dimension]), item_embeddings[1:, :]], 0)
			users = tf.nn.embedding_lookup(user_embeddings, batch_users)
			sequences = tf.nn.embedding_lookup(item_embeddings, batch_sequences)

		# feature gating layer
		with tf.variable_scope("feature_gating_layer", reuse = tf.AUTO_REUSE):
			w1 = tf.get_variable("w1", shape = [self.latent_dimension, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			w2 = tf.get_variable("w2", shape = [self.latent_dimension, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			b = tf.get_variable("b", shape = [self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))

			seq_w1 = tf.reshape(tf.matmul(tf.reshape(sequences, [-1, self.latent_dimension]), w1), [-1, self.sequence_length, self.latent_dimension])
			user_w2 = tf.matmul(users, w2)
			fea_gate = tf.sigmoid(seq_w1 + tf.expand_dims(user_w2, 1) + b)
			sequences_f = sequences * fea_gate

		# instance gating layer
		with tf.variable_scope("instance_gating_layer", reuse = tf.AUTO_REUSE):
			w3 = tf.get_variable("w3", shape = [self.latent_dimension, 1], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			w4 = tf.get_variable("w4", shape = [self.latent_dimension, self.sequence_length], initializer = tf.truncated_normal_initializer(stddev = 0.01))

			seq_w3 = tf.matmul(sequences_f, tf.tile(tf.expand_dims(w3, 0), [tf.shape(sequences_f)[0], 1, 1]))
			user_w4 = tf.matmul(users, w4)
			ins_gate = tf.sigmoid(tf.squeeze(seq_w3, -1) + user_w4)
			sequences_i = sequences_f * tf.expand_dims(ins_gate, -1)

		# aggregation layer
		sequences_i = tf.reduce_mean(sequences_i, 1)
		if is_training: sequences_i = tf.nn.dropout(sequences_i, self.keep_prob)

		# fc layer
		with tf.variable_scope("fc_layer", reuse = tf.AUTO_REUSE):
			weights = tf.get_variable("weights", shape = [self.item_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			pred_weights = tf.nn.embedding_lookup(weights, batch_items)
			pred_logits = tf.squeeze(tf.matmul(pred_weights, tf.expand_dims(users, -1)), -1)
			pred_logits += tf.squeeze(tf.matmul(pred_weights, tf.expand_dims(sequences_i, -1)), -1)
			pred_logits += tf.reduce_sum(tf.matmul(sequences, tf.transpose(pred_weights, [0,2,1])),1)
				
			# if train, return loss and train_op; if test, return predictions
			if is_training:
				tar_logits, neg_logits = pred_logits[:, :self.target_length], pred_logits[:, self.target_length:]
				loss = tf.reduce_mean(-tf.log(tf.sigmoid(tar_logits - neg_logits) + 1e-8))
				train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)
				return loss, train_op
			else:
				predictions = tf.nn.softmax(pred_logits)
				return predictions
			
