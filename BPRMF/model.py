import tensorflow as tf



class Model():
	def __init__(self, args, data):
		# initialize parameters
		self.learn_rate = args.learn_rate
		self.latent_dimension = args.latent_dimension

		self.user_size = data.user_size
		self.item_size = data.item_size


		
	def forward(self, batch_triples, batch_users, batch_items, is_training):	
		# embedding layer
		with tf.variable_scope("embedding_layer", reuse = tf.AUTO_REUSE):
			user_embeddings = tf.get_variable("user_embeddings", shape = [self.user_size, self.latent_dimension], 
								initializer = tf.truncated_normal_initializer(stddev = 0.01))
			item_embeddings = tf.get_variable("item_embeddings", shape = [self.item_size, self.latent_dimension], 
								initializer = tf.truncated_normal_initializer(stddev = 0.01))
		
		if is_training:
			u, i, j = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
			u_emb = tf.nn.embedding_lookup(user_embeddings, u)
			i_emb = tf.nn.embedding_lookup(item_embeddings, i)
			j_emb = tf.nn.embedding_lookup(item_embeddings, j)

			uij =  tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1)
			loss = -tf.reduce_mean(tf.log(tf.sigmoid(uij)))
			train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)
			return loss, train_op
		else:
			user_emb = tf.nn.embedding_lookup(user_embeddings, batch_users)
			item_emb = tf.nn.embedding_lookup(item_embeddings, batch_items)
			logits = tf.squeeze(tf.matmul(item_emb, tf.expand_dims(user_emb, -1)), -1)
			predictions = tf.nn.softmax(logits)
			return predictions

