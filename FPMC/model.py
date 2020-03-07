import tensorflow as tf


class Model():
	def __init__(self, args, data):
		# initialize parameters
		self.sequence_length = args.sequence_length
		self.learn_rate = args.learn_rate
		self.latent_dimension = args.latent_dimension

		self.user_size = data.user_size
		self.item_size = data.item_size


	def forward(self, batch_users, batch_sequences, batch_items, is_training):	
		with tf.variable_scope("embedding_layer", reuse = tf.AUTO_REUSE):
			self.VUI = tf.get_variable("VUI", shape = [self.user_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			self.VIU = tf.get_variable("VIU", shape = [self.item_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			self.VIL = tf.get_variable("VIL", shape = [self.item_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))
			self.VLI = tf.get_variable("VLI", shape = [self.item_size, self.latent_dimension], initializer = tf.truncated_normal_initializer(stddev = 0.01))

		if is_training:
			batch_targets, batch_negatives = batch_items[:, 0], batch_items[:, 1]

			users = tf.nn.embedding_lookup(self.VUI, batch_users)
			targets1 = tf.nn.embedding_lookup(self.VIU, batch_targets)
			targets2 = tf.nn.embedding_lookup(self.VIL, batch_targets)
			negatives1 = tf.nn.embedding_lookup(self.VIU, batch_negatives)
			negatives2 = tf.nn.embedding_lookup(self.VIL, batch_negatives)
			sequences = tf.nn.embedding_lookup(self.VLI, batch_sequences)

			targets2 = tf.tile(tf.expand_dims(targets2, 1), [1, self.sequence_length, 1])
			negatives2 = tf.tile(tf.expand_dims(negatives2, 1), [1, self.sequence_length, 1])

			pos_score1 = tf.reduce_sum(users*targets1, 1)
			pos_score2 = tf.reduce_mean(tf.reduce_sum(targets2*sequences, 2), 1)
			pos_score = pos_score1 + pos_score2

			neg_score1 = tf.reduce_sum(users*negatives1, 1)
			neg_score2 = tf.reduce_mean(tf.reduce_sum(negatives2*sequences, 2), 1)
			neg_score = neg_score1 + neg_score2

			loss = -tf.reduce_mean(tf.log(tf.sigmoid(pos_score - neg_score)))	
			train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)
			return loss, train_op
		else:
			user_emb = tf.nn.embedding_lookup(self.VUI, batch_users)
			pred_emb1 = tf.nn.embedding_lookup(self.VIU, batch_items)
			score1 = tf.squeeze(tf.matmul(pred_emb1, tf.expand_dims(user_emb, -1)), -1)

			seq_emb = tf.nn.embedding_lookup(self.VLI, batch_sequences)
			pred_emb2 = tf.nn.embedding_lookup(self.VIL, batch_items)
			score2 = tf.reduce_mean(tf.matmul(seq_emb, tf.transpose(pred_emb2, [0,2,1])), 1)

			score = score1 + score2
			predictions = tf.nn.softmax(score)
			return predictions
