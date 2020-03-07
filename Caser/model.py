import tensorflow as tf



class Model():
	def __init__(self, args, data):
		# initialize parameters
		self.L = args.sequence_length
		self.T = args.target_length
		self.hn = args.horizontal_filter_num
		self.vn = args.vertical_filter_num
		self.learn_rate = args.learn_rate
		self.keep_prob = args.keep_prob
		self.d = args.latent_dimension

		self.user_size = data.user_size
		self.item_size = data.item_size


	def forward(self, batch_users, batch_sequences, batch_items, is_training):	
		# embedding layer
		with tf.variable_scope("embedding_layer", reuse = tf.AUTO_REUSE):
			user_embeddings = tf.get_variable("user_embeddings", shape = [self.user_size, self.d], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			item_embeddings = tf.get_variable("item_embeddings", shape = [self.item_size, self.d], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			item_embeddings = tf.concat([tf.zeros(shape=[1, self.d]), item_embeddings[1:, :]], 0)
			users = tf.nn.embedding_lookup(user_embeddings, batch_users)
			sequences = tf.nn.embedding_lookup(item_embeddings, batch_sequences)
			sequences = tf.expand_dims(sequences, axis = -1)

		# horizontal convolutional layer
		with tf.variable_scope("h_conv_layer", reuse = tf.AUTO_REUSE):
			weights = []
			for i in range(self.L):
				weight = tf.get_variable("weight_%d" % (i+1), shape = [i+1, self.d, 1, self.hn], initializer = tf.truncated_normal_initializer(stddev = 0.01))
				weights.append(weight)

			h_out = [tf.nn.conv2d(sequences, weight, strides = [1, 1, 1, 1], padding = 'VALID') for weight in weights]
			h_out = [tf.nn.relu(out) for out in h_out]
			h_out = [tf.reduce_max(out, axis = 1, keepdims = True) for out in h_out]
			h_out = tf.concat(h_out, 1)
			h_out = tf.reshape(h_out, shape = [-1, self.hn*self.L])

		# vertical convolutional layer
		with tf.variable_scope("v_conv_layer", reuse = tf.AUTO_REUSE):
			weight = tf.get_variable("weight", shape = [self.L, 1, 1, self.vn], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			v_out = tf.nn.conv2d(sequences, weight, strides = [1, 1, 1, 1], padding = 'VALID')
			v_out = tf.reshape(v_out, shape = [-1, self.vn*self.d])
			
		# concat the horizontal and vertical output
		out = tf.concat([h_out, v_out], 1)
		if is_training: out = tf.nn.dropout(out, self.keep_prob)
		
		# the 1th full connection layer
		with tf.variable_scope("fc_layer1", reuse = tf.AUTO_REUSE):
			weight = tf.get_variable("weight", shape = [self.hn*self.L + self.vn*self.d, self.d], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			bias = tf.get_variable("bias", shape = [self.d], initializer = tf.zeros_initializer())
			fc1_out = tf.nn.relu(tf.matmul(out, weight) + bias)
		
		# aggregate item embedding and user embedding
		fc2_in = tf.concat([fc1_out, users], 1)

		# the 2th full connection layer
		with tf.variable_scope("fc_layer2", reuse = tf.AUTO_REUSE): 
			weight = tf.get_variable("weight", shape = [self.item_size, self.d+self.d], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			bias = tf.get_variable("bias", shape = [self.item_size], initializer = tf.zeros_initializer())
			pred_weight = tf.nn.embedding_lookup(weight, batch_items)
			pred_bias = tf.nn.embedding_lookup(bias, batch_items)
			pred_logits = tf.squeeze(tf.matmul(pred_weight, tf.expand_dims(fc2_in, -1)), -1) + pred_bias
				
			# if train, return loss and train_op; if test, return predictions
			if is_training:
				tar_logits, neg_logits = pred_logits[:, :self.T], pred_logits[:, self.T:]
				tar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(tar_logits) + 1e-8))
				neg_loss = tf.reduce_mean(-tf.log(1-tf.sigmoid(neg_logits) + 1e-8))

				loss = tar_loss + neg_loss
				train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)
				return loss, train_op
			else:
				predictions = tf.nn.softmax(pred_logits)
				return predictions
			
