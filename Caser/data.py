import pandas as pd
import numpy as np
import random
import time
from util import *



class Data:
	def __init__(self, args):
		# initialize parameters
		self.file_path = 'files/%s/%s.csv' % (args.dataset, args.dataset)
		self.sequence_length = args.sequence_length
		self.target_length = args.target_length
		self.sample_ratio = args.sample_ratio
		
		self.union_length = self.sequence_length + self.target_length
		self.negative_length = self.target_length * self.sample_ratio
		
		# read data from file, sort by time for each user
		self.data = pd.read_csv(self.file_path, header=0)
		self.data = self.data.sort_values(by = ['user', 'time'])
		
		# map original user id to new user id(0,1,2...)
		self.user_map, new_id = {}, 0	
		user_set = set(self.data['user'])
		
		for u in user_set:
			self.user_map[u] = new_id
			new_id += 1				
			
		self.data['user'] = self.data['user'].map(lambda x: self.user_map[x])
		self.user_set = set(self.data['user'])
		self.user_size = len(self.user_set)

		# map original item id to new item id(1,2,3...), 0 is for padding the sequence		
		self.item_map, new_id = {}, 1	
		item_set = set(self.data['item'])
		
		for i in item_set:
			self.item_map[i] = new_id
			new_id += 1				
			
		self.data['item'] = self.data['item'].map(lambda x: self.item_map[x])
		self.item_set = set(self.data['item'])
		self.item_size = len(self.item_set) + 1
		
		# begin to prepare data
		print('\nprepare data begin...')
		start_time = time.time()
		self.split_dataset()
		self.get_input_data()
		used_time = (time.time() - start_time) / 60
		print('prepare data end, time used: %.2f min.\n' % used_time)



	def split_dataset(self):
		self.train_set, self.valid_set, self.test_set = {}, {}, {}
		user_list, item_list = list(self.data['user']), list(self.data['item'])
		user_ids, indices, counts = np.unique(user_list, return_index = True, return_counts = True)

		for i in range(len(user_ids)):
			uid, index, length = user_ids[i], indices[i], counts[i]
			items = item_list[index: index+length]
			self.train_set[uid], self.valid_set[uid], self.test_set[uid] = items[:-2], [items[-2]], [items[-1]]
		


	def get_input_data(self):
		self.train_users, self.train_sequences, self.train_items = [], [], []
		self.valid_users, self.valid_sequences, self.valid_items = [], [], []
		self.test_users, self.test_sequences, self.test_items = [], [], []
		valid_test_num = 0

		for u in np.random.permutation(self.user_size):
			rated = self.train_set[u]
			sample_pool = list(self.item_set - set(rated))

			# train data
			for seq in slide_window(self.train_set[u], self.union_length):
				self.train_users.append(u)
				self.train_sequences.append(seq[:self.sequence_length])
				self.train_items.append(seq[-self.target_length:] + random.sample(sample_pool, self.negative_length))

			if valid_test_num >= 10000: continue
			valid_test_num += 1

			# valid data
			self.valid_users.append(u)
			self.valid_sequences.append(get_recent(self.train_set[u], self.sequence_length))
			self.valid_items.append(self.valid_set[u] + random.sample(sample_pool, 500))
		
			# test data
			self.test_users.append(u)
			self.test_sequences.append(get_recent(self.train_set[u], self.sequence_length-1) + self.valid_set[u])
			self.test_items.append(self.test_set[u] + random.sample(sample_pool, 500))

