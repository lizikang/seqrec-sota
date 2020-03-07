from data import Data
from model import Model
from util import *
import tensorflow as tf
import argparse
import time



def main():
	# define agruments, data and model
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', required=True)
	parser.add_argument('--sequence_length', type=int, default=5)
	parser.add_argument('--layer_num', type=int, default=1)
	parser.add_argument('--neg_samples', type=int, default=3)

	parser.add_argument('--epoch_num', type=int, default=400)
	parser.add_argument('--batch_size', type=int, default=1024)
	parser.add_argument('--learn_rate', type=float, default=1e-3)
	parser.add_argument('--latent_dimension', type=int, default=100)
	parser.add_argument('--logid', type=int, required=True)

	args = parser.parse_args()
	print(str(args))
	data = Data(args)
	model = Model(args, data)
	train(args, data, model)



def train(args, data, model):
	# define placeholder, loss, train_op and predictions
	batch_sequences = tf.placeholder(tf.int32, [None, args.sequence_length])
	batch_items = tf.placeholder(tf.int32, [None, None])
	loss, train_op = model.forward(batch_sequences, batch_items, True)
	predictions = model.forward(batch_sequences, batch_items, False)

	# create session to run model
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		train_iterations = len(data.train_sequences) // args.batch_size + 1
		bmap_, bhr_5, bhr_10, bndcg_5, bndcg_10, bepoch, btest_preds, times = 0, 0, 0, 0, 0, 0, [], []

		for i in range(1, args.epoch_num+1):
			# compute and display train performance
			start_time = time.time()
			for j in range(1, train_iterations+1):
				start, end = args.batch_size * (j-1), args.batch_size * j
				train_feed_dict = {batch_sequences: data.train_sequences[start:end], 
							batch_items: data.train_items[start:end]}
				loss_, train_op_ = sess.run([loss, train_op], feed_dict = train_feed_dict)
				if j % (train_iterations//3) == 0: 
					print('epoch: %d, iteration: %d, loss: %f' % (i, j, loss_))
			times.append(time.time() - start_time)

			if i % 1 == 0:
				# compute valid performance
				valid_feed_dict = {batch_sequences: data.valid_sequences, batch_items: data.valid_items}
				valid_preds = sess.run(predictions, feed_dict = valid_feed_dict)

				rank = (-valid_preds).argsort(-1).argsort(-1)[:, 0]
				map_ = map(rank); hr_5 = hr_k(rank, 5); hr_10 = hr_k(rank, 10); ndcg_5 = ndcg_k(rank, 5); ndcg_10 = ndcg_k(rank, 10)
				print('epoch: %d, valid map: %.4f, hr@5: %.4f, hr@10: %.4f, ndcg@5: %.4f, ndcg@10: %.4f\n' % (i, map_, hr_5, hr_10, ndcg_5, ndcg_10))

				# compute test performance
				test_feed_dict = {batch_sequences: data.test_sequences, batch_items: data.test_items}
				test_preds = sess.run(predictions, feed_dict = test_feed_dict)

				# record the best validation performance 
				if int(map_ >= bmap_) + int(hr_5 >= bhr_5) + int(hr_10 >= bhr_10) + int(ndcg_5 >= bndcg_5) + int(ndcg_10 >= bndcg_10) >= 4:
					bmap_, bhr_5, bhr_10, bndcg_5, bndcg_10, bepoch, btest_preds = map_, hr_5, hr_10, ndcg_5, ndcg_10, i, test_preds

				if i - bepoch >= 20:
					# calculate time and corresponding test performance
					total_time = sum(times[:bepoch]) / 60
					avg_time = np.mean(times[:bepoch]) / 60
					rank = (-btest_preds).argsort(-1).argsort(-1)[:, 0]
					map_ = map(rank); hr_5 = hr_k(rank, 5); hr_10 = hr_k(rank, 10); ndcg_5 = ndcg_k(rank, 5); ndcg_10 = ndcg_k(rank, 10)

					print(args.logid, ':', args.sequence_length, args.layer_num, args.neg_samples, '\t', args.epoch_num, args.batch_size, args.learn_rate, args.latent_dimension)
					print('epoch: %d, time: %.2f min, %.2f min, (valid) map: %.4f, hr@5: %.4f, hr@10: %.4f, ndcg@5: %.4f, ndcg@10: %.4f, (test) map: %.4f, hr@5: %.4f, hr@10: %.4f, ndcg@5: %.4f, ndcg@10: %.4f' % 
						(bepoch, total_time, avg_time, bmap_, bhr_5, bhr_10, bndcg_5, bndcg_10, map_, hr_5, hr_10, ndcg_5, ndcg_10))
					break
					


if __name__ == '__main__':
	main()
