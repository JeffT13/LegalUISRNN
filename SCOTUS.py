''' SCOTUS d-vec UISRNN processing'''

import numpy as np
import torch
import uisrnn

'''
Things to look into:
	- loading training/testing data in
	- args?
	- multiple .fit() calls
	- input concatenation
	- labels to string
'''

#load data
train_seqs = []
train_labels = []
test_items = []

train_case_path = './data/SCOTUS_Processed/train/*'
test_case_path = './data/SCOTUS_Processed/test/*'

#load training data
for case in enumerate(train_case_path):
	case_id = case.split('/')
	temp_seq = np.load(case+'/'+case_id+'_emb.npy', allow_pickle=True)
	temp_lab = np.load(case+'/'+case_id+'_label.npy', allow_pickle=True)
	train_seq.append(temp_seq)
	train_labels.append(temp_lab)
	

#Define UISRNN
model_args, training_args, inference_args = uisrnn.parse_arguments()
model = uisrnn.UISRNN(args)

#model training **MUST BE LOOP TO MANAGE SEQ LENGTH**
model.fit(train_sequences, train_cluster_ids, args, enforce_cluster_id_uniqueness=False)


#evaluation has similar mechanic
predicted_cluster_ids = model.predict(test_sequences, args)

