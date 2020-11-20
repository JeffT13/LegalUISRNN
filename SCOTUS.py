''' SCOTUS d-vec UISRNN processing'''

import numpy as np
import torch
import glob
import os
import uisrnn



'''
Things to look into:
	- arg:

        MODEL
        = estimatable:
                - transition_bias (eq 13) (prob should estimate)
                - sigma2 (eq 11)
        = crp_alpha (eq 7, *cannot be estimated)
        TRAINING
        = batch_size?
	= num_permutations?

	- multiple .fit() calls & input concatenation
	- labels to string
'''

#load data
train_seqs = []
train_labels = []
test_items = []

train_case_path = './data/SCOTUS_Processed/train/*/*'
#test_case_path = './data/SCOTUS_Processed/test/*'

case_path = glob.glob(os.path.dirname(train_case_path))

print(case_path)
#load training data

train_sequences = []
train_cluster_ids = []
for i, case in enumerate(case_path):
	print(case)
	case_id = case.split('/')[-1][:-7]
	print(case_id)
	temp_seq = np.load(case+'/'+case_id+'_emb.npy', allow_pickle=True)
	temp_lab = np.load(case+'/'+case_id+'_label.npy', allow_pickle=True)
	if True:
		print(np.shape(temp_seq))
		print(np.shape(temp_lab))
	train_sequences.append(temp_seq)
	train_cluster_ids.append(temp_lab)
	break


#Define UISRNN
model_args, training_args, inference_args = uisrnn.parse_arguments()
model_args.verbosity=3
model_args.observation_dim=256 #from hparam techincally

training_args.enforce_cluster_id_uniqueness=False #based on dvec_SCOTUS
training_args.batch_size = 3

#model training **MUST BE LOOP TO MANAGE SEQ LENGTH**
model = uisrnn.UISRNN(model_args)
model.fit(train_sequences, train_cluster_ids, training_args)

#evaluation has similar mechanic
#predicted_cluster_ids = model.predict(test_sequences, inference_args)

