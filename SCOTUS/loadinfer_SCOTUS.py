''' SCOTUS d-vec UISRNN processing'''

import sys
sys.path.append("./LegalUISRNN")
import numpy as np
import torch
import glob
import os
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

#expects processed cases in data folder (take from Google Drive or PRINCE)
case_path = '/scratch/jt2565/SCOTUS_Processed/*/*'
case_path = glob.glob(os.path.dirname(case_path))

total_cases = len(case_path)
train_cases = total_cases//10*8
print("# of training:", train_cases)
print("# total cases:" , total_cases)

train_sequences = []
train_cluster_ids = []
test_sequences = []
test_cluster_ids = []

verbose = True
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
           
    if i <= train_cases:
        trn_seq_lst.append(train_sequence)
        trn_cluster_lst.append(train_cluster_id)
    else:
        test_seq_lst.append(train_sequence)
        test_cluster_lst.append(train_cluster_id) 

#Define UISRNN
model_args, training_args, inference_args = uisrnn.parse_arguments()
model_args.verbosity=3
model_args.observation_dim=256 #from hparam
model_args.enable_cuda = True
model_args.rnn_depth = 2
model_args.rnn_hidden_size = 8
inference_args.test_iteration = 2 


# attempt to save model
model.load('./localsamp_uisrnn.pth')

print('-'*10, 'testing')
for c in range(len(trn_seq_lst))
    test_sequences = test_seq_lst[c]
    test_cluster_ids = test_cluster_lst[c]
    if verbose:
        print('testing case', c)
        
    #evaluation has similar mechanic
    predicted_cluster_ids = model.predict(test_sequences, inference_args)
    print(type(predicted_cluster_ids))
    print('='*50)
    model.logger.print(3, 'Asserting the equivalence between \nGround truth: {}\nPredicted: {}'.format(test_cluster_id, predicted_label))
    print('Asserting the equivalence between','\nGround truth: {}\nPredicted: {}'.format(test_cluster_id, predicted_label))
    accuracy = uisrnn.compute_sequence_match_accuracy(predicted_label, test_cluster_id)
    print('acc:', accuracy)


