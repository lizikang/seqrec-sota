import tensorflow as tf
import numpy as np
import argparse
import time

import model
from data import Data
from util import *



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--beta1', type = float, default = 0.9, help = 'hyperpara-Adam')
	parser.add_argument('--percentage', type = float, default = 0.8, help='0.8 means 80% training 20% testing')
	parser.add_argument('--is_generatesubsession', type = bool, default = False, help = 'whether generating subsessions')

	parser.add_argument('--dataset', required=True)
	parser.add_argument('--sequence_length', type=int, default=5)

	parser.add_argument('--epoch_num', type=int, default=400)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--learn_rate', type=float, default=1e-3)
	parser.add_argument('--dilated_channels', type=int, default=100)
	parser.add_argument('--logid', type=int, required=True)
	args = parser.parse_args()
	print(str(args))

	# load data from file
	data = Data(args)	
	item_size = data.item_size
	train_sequences, test_sequences = data.train_sequences, data.test_sequences

	# generate subsession
	if args.is_generatesubsession:
		train_set = generate_subsequences(train_set)
		
	model_para = {
		'item_size': item_size,
		# if you use nextitnet_residual_block, you can use [1, 4, ],
		# if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
		'dilations': [1, 2],
		'kernel_size': 3,

		'epoch_num': args.epoch_num,
		'batch_size': args.batch_size,  
		'learn_rate': args.learn_rate, 
		'dilated_channels': args.dilated_channels,
		'is_negsample':False
	}

	itemrec = model.NextItNet_Decoder(model_para)
	itemrec.train_graph(model_para['is_negsample'])
	optimizer = tf.train.AdamOptimizer(model_para['learn_rate'], beta1 = args.beta1).minimize(itemrec.loss)
	itemrec.predict_graph(model_para['is_negsample'], reuse = True)

	# create session to run model
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		train_iterations = len(data.train_sequences) // model_para['batch_size'] + 1
		bmap_, bhr_5, bhr_10, bndcg_5, bndcg_10, bepoch, btest_preds, times = 0, 0, 0, 0, 0, 0, [], []

		for i in range(1, model_para['epoch_num']+1):
			# compute and display train performance
			start_time = time.time()
			for j in range(1, train_iterations+1):
				start, end = model_para['batch_size'] * (j-1), model_para['batch_size'] * j
				train_feed_dict = {itemrec.itemseq_input: data.train_sequences[start:end]}
				_, loss_ = sess.run([optimizer, itemrec.loss], feed_dict = train_feed_dict)
				if j % (train_iterations//3) == 0: 
					print('epoch: %d, iteration: %d, loss: %f' % (i, j, loss_))
			times.append(time.time() - start_time)

			if i % 1 == 0:
				# compute valid performance
				valid_feed_dict = {itemrec.input_predict: data.valid_sequences}
				valid_logits = sess.run(itemrec.logits, feed_dict = valid_feed_dict)

				row_indices = np.arange(len(data.valid_sequences))[:, np.newaxis]
				col_indices = np.array(data.valid_items)
				valid_logits = valid_logits[row_indices, col_indices]
				valid_preds = valid_logits

				rank = (-valid_preds).argsort(-1).argsort(-1)[:, 0]
				map_ = map(rank); hr_5 = hr_k(rank, 5); hr_10 = hr_k(rank, 10); ndcg_5 = ndcg_k(rank, 5); ndcg_10 = ndcg_k(rank, 10)
				print('epoch: %d, valid map: %.4f, hr@5: %.4f, hr@10: %.4f, ndcg@5: %.4f, ndcg@10: %.4f\n' % (i, map_, hr_5, hr_10, ndcg_5, ndcg_10))

				# compute test performance
				test_feed_dict = {itemrec.input_predict: data.test_sequences}
				test_logits = sess.run(itemrec.logits, feed_dict = test_feed_dict)

				row_indices = np.arange(len(data.test_sequences))[:, np.newaxis]
				col_indices = np.array(data.test_items)
				test_logits = test_logits[row_indices, col_indices]
				test_preds = test_logits

				# record the best validation performance 
				if int(map_ >= bmap_) + int(hr_5 >= bhr_5) + int(hr_10 >= bhr_10) + int(ndcg_5 >= bndcg_5) + int(ndcg_10 >= bndcg_10) >= 4:
					bmap_, bhr_5, bhr_10, bndcg_5, bndcg_10, bepoch, btest_preds = map_, hr_5, hr_10, ndcg_5, ndcg_10, i, test_preds

				if i - bepoch >= 30:
					# calculate time and corresponding test performance
					total_time = sum(times[:bepoch]) / 60
					avg_time = np.mean(times[:bepoch]) / 60
					rank = (-btest_preds).argsort(-1).argsort(-1)[:, 0]
					map_ = map(rank); hr_5 = hr_k(rank, 5); hr_10 = hr_k(rank, 10); ndcg_5 = ndcg_k(rank, 5); ndcg_10 = ndcg_k(rank, 10)

					print(args.logid, ':', args.sequence_length, model_para['dilations'], model_para['kernel_size'], '\t', 
						args.epoch_num, args.batch_size, args.learn_rate, args.dilated_channels)
					print('epoch: %d, time: %.2f min, %.2f min, (valid) map: %.4f, hr@5: %.4f, hr@10: %.4f, ndcg@5: %.4f, ndcg@10: %.4f, (test) map: %.4f, hr@5: %.4f, hr@10: %.4f, ndcg@5: %.4f, ndcg@10: %.4f' % 
						(bepoch, total_time, avg_time, bmap_, bhr_5, bhr_10, bndcg_5, bndcg_10, map_, hr_5, hr_10, ndcg_5, ndcg_10))
					break

	
	

def generate_subsequences(train_set, ratio = 3):
	subsequences = []
	
	for seq in train_set:
		for i in range(ratio):
			beg = [0] * i
			end = seq[:-i] if i != 0 else seq
			beg.extend(end)
			subsequences.append(beg)
			
	np.random.shuffle(subsequences)
	return np.array(subsequences)
	

	
if __name__ == '__main__':
	main()
