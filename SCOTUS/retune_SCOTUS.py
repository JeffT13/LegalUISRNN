''' SCOTUS d-vec UISRNN processing'''

import sys
import os
import csv
import numpy as np
import glob
import torch

sys.path.append("./LegalUISRNN")
import uisrnn



'''
Things to look into:
	- arg:
        = estimatable:
                - transition_bias (eq 13) (prob should estimate)
                - sigma2 (eq 11)
        = crp_alpha (eq 7, *cannot be estimated)
	= num_permutations?
	- multiple .fit() calls & input concatenation
'''

#expects processed cases in data folder (take from Google Drive or PRINCE)
#case_path = './LegalUISRNN/data/SCOTUS_Processed/*/*'  # local path
case_path = '/scratch/jt2565/SCOTUS_Processed/*/*'      # prince path

case_path = glob.glob(os.path.dirname(case_path))

total_cases = len(case_path)
train_cases = total_cases//10*9
print("# of training:", train_cases)
print("# total cases:" , total_cases)

trn_seq_lst = []
trn_cluster_lst = []
test_seq_lst = []
test_cluster_lst = []
test_case_lst = []

verbose = False #false for tuning
flatten = True
for i, case in enumerate(case_path):

    case_id = case.split('/')[-1][:-7]
    train_seq = np.load(case+'/'+case_id+'_sequence.npy', allow_pickle=True)
    train_clus = np.load(case+'/'+case_id+'_cluster_id.npy', allow_pickle=True)

    train_sequence = []
    train_cluster_id = []
    for j in range(np.shape(train_seq)[0]):
        train_sequence.append(train_seq[j])
        if i <= train_cases:
            train_cluster_id.append(train_clus[j])
        else:
            train_cluster_id.append(list(map(int, train_clus[j])))
               
    if verbose:
        print('Processed case:', case_id)
        print('emb shape:', np.shape(train_seq))
        print('label shape:', np.shape(train_clus))
        print('emb len:', len(train_sequence))
        print('label len:', len(train_cluster_id))    
    

    if flatten:
        train_sequence  = np.concatenate(train_sequence, axis = 0)
        train_cluster_id = np.concatenate(train_cluster_id, axis = 0)
    if i <= train_cases:
        trn_seq_lst.append(train_sequence)
        trn_cluster_lst.append(train_cluster_id)
    else:        
        test_case_lst.append(case.split('/')[-1])
        test_seq_lst.append(train_sequence)
        test_cluster_lst.append(train_cluster_id) 


#Define UISRNN
model_args, training_args, inference_args = uisrnn.parse_arguments()
model_args.verbosity=3 #prints model performance
model_args.observation_dim=256 #from hparam
model_args.enable_cuda = True
model_args.rnn_depth = 2
model_args.rnn_hidden_size = 64
training_args.learning_rate = 0.01
training_args.train_iteration = 500
training_args.enforce_cluster_id_uniqueness=False #based on dvec_SCOTUS
training_args.batch_size = 5
model = uisrnn.UISRNN(model_args)

epochs = 5

if flatten:
    training_args.train_iteration = 400
    print('-'*10, 'full seq training for ', len(trn_seq_lst), 'cases')
    for e in range(epochs):
        print('='*10, 'EPOCH ', e, '='*10)
        model.fit(trn_seq_lst, trn_cluster_lst, training_args)
    print('-'*10, 'training complete')
    
else:
    training_args.train_iteration = 40
    print('-'*10, 'per case training for ', len(trn_seq_lst), 'cases')

    for e in range(epochs):
        print('='*10, 'EPOCH ', e, '='*10)
        for c in range(len(trn_seq_lst)):
            train_sequences = trn_seq_lst[c]
            train_cluster_ids = trn_cluster_lst[c]
            if verbose: #off for tuning
                print('training case', c)
                print('list?', isinstance(train_sequences, list))
                print('item type?', type(train_sequences[0]))
                print('num utt', len(train_sequences))

            model.fit(train_sequences, train_cluster_ids, training_args)
    print('-'*10, 'training complete')

    
# attempt to save model
model.save('./princetune_uisrnn.pth')  
print('model saved')

with open('./uisrnn_testcases.csv', 'w') as rm:
    wr = csv.writer(rm, delimiter="\n")
    wr.writerow(test_case_lst)
