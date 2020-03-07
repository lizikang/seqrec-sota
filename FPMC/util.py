import tensorflow as tf
import numpy as np
import copy



def map(pred_rank):
	ap =  1 / (pred_rank + 1)
	return np.mean(ap)



def hr_k(pred_rank, k):
	rank = copy.deepcopy(pred_rank)
	rank[rank<k] = 1
	rank[rank>=k] = 0
	hr = np.mean(rank)
	return hr
	


def ndcg_k(pred_rank, k):
	rank = copy.deepcopy(pred_rank)
	rank = rank.astype(np.float64)
	rank[rank>=k] = np.inf
	ndcg = 1 / np.log2(rank+2)
	ndcg = np.mean(ndcg)
	return ndcg



def slide_window(sequence, window_size, step_size=1):
	sequence_length = len(sequence)

	if sequence_length >= window_size:
		for i in range(0, sequence_length-window_size+1, step_size):
				yield sequence[i: i+window_size]
	else:
		padding_size = window_size - sequence_length
		yield list(np.pad(sequence, (padding_size, 0), 'constant'))



def get_recent(sequence, k):
	sequence_length = len(sequence)
	if sequence_length >= k:
		return sequence[-k:]
	else:
		return list(np.pad(sequence, (k-sequence_length, 0), 'constant'))

