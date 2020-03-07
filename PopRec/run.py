from data import Data
from collections import defaultdict
import argparse
import numpy as np



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True)
	args = parser.parse_args()

	print(str(args))
	data = Data(args)
	run(data)



def run(data):
	# calculate the popularity of items
	item_dict = defaultdict(lambda: 0)
	for i in data.train_items:
		item_dict[i] += 1

	# calculate the test performance of each user
	ap, hr_5, hr_10, ndcg_5, ndcg_10 = 0.0, 0.0, 0.0, 0.0, 0.0
	for pred_items in data.pred_items:
		pred_times = list(map(lambda x: item_dict[x], pred_items))
		rank = (-np.array(pred_times)).argsort().argsort()[0]
		
		ap += 1 / (rank+1)
		if rank < 5:
			hr_5 += 1
			ndcg_5 += 1 / np.log2(rank + 2)
		if rank < 10:
			hr_10 += 1
			ndcg_10 += 1 / np.log2(rank + 2)

	# calculate and dislpay the average performance of all users
	map_ = ap / data.pred_num
	hr_5 = hr_5 / data.pred_num
	hr_10 = hr_10 / data.pred_num
	ndcg_5 = ndcg_5 / data.pred_num
	ndcg_10 = ndcg_10 / data.pred_num
	print('(test) map: %.4f, hr@5: %.4f, hr@10: %.4f, ndcg@5: %.4f, ndcg@10: %.4f' % (map_, hr_5, hr_10, ndcg_5, ndcg_10))



if __name__ == '__main__':
	main()
