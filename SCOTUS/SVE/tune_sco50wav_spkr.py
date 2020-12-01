''' SCOTUS SVE case-embed dvec -- UISRNN processing'''

import sys
sys.path.append("./LegalUISRNN")
import glob
import numpy as np
import torch
import os
import uisrnn

spkr_path = '/scratch/jt2565/sco50/sco50wav_proc_spkr/*/*'      # prince path
spkr_path = glob.glob(os.path.dirname(spkr_path))

total_cases = len(spkr_path)
train_cases = (total_cases//10)*9
print("# of training:", train_cases)
print("# total cases:" , total_cases)

trn_seq_lst = []
trn_cluster_lst = []
test_seq_lst = []
test_cluster_lst = []

verbose = False

if verbose:
    print("\n", "="*50, "\n Processing spkr-embedded d-vec")
        
#load 5 case-embedded dvecs (with directory holding raw files)
for i, casespkr in enumerate(spkr_path):

    case_id = casespkr.split('/')[-1][:-7]
    train_seq = np.load(casespkr+'/'+case_id+'_sequence.npy',allow_pickle=True)
    train_clus = np.load(casespkr+'/'+case_id+'_cluster_id.npy', allow_pickle=True)
    train_sequence = []
    train_cluster_id = []
    
    #converts labels to int for inference/testing
    for j in range(np.shape(train_clus)[0]):
        train_sequence.append(train_seq[j])
        if i <= train_cases:
            train_cluster_id.append(train_clus[j])
        else:
            train_cluster_id.append(list(map(int, train_clus[j])))
       
    train_sequence  = np.concatenate(train_sequence, axis = 0)
    train_cluster_id = np.concatenate(train_cluster_id, axis = 0)   

    if verbose:
        if i > train_cases:
            print("-- Stored as test case --")
        else:
            print("-- Stored as train case --")
        print('Processed case:', case_id)
        print('emb shape:', np.shape(train_seq))
        print('label shape:', np.shape(train_clus))
        print('flat emb:', np.shape(train_sequence))
        print('flat label:', np.shape(train_cluster_id))      
    if i <= train_cases:
        trn_seq_lst.append(train_sequence)
        trn_cluster_lst.append(train_cluster_id)
    else:
        test_seq_lst.append(train_sequence)
        test_cluster_lst.append(train_cluster_id) 
        
        
        
    
#Define UISRNN
model_args, training_args, inference_args = uisrnn.parse_arguments()
model_args.verbosity=3 #can verbose=False for no prints except training
model_args.observation_dim=256 #from hparam
model_args.enable_cuda = True
model_args.rnn_depth = 2
model_args.rnn_hidden_size = 64
training_args.learning_rate = 0.01
training_args.train_iteration = 1000
training_args.enforce_cluster_id_uniqueness=False #based on dvec_SCOTUS
training_args.batch_size = 10
model = uisrnn.UISRNN(model_args)


#TRAIN
model.fit(trn_seq_lst, trn_cluster_lst, training_args)

'''
print('-'*10, 'full seq training for ', len(trn_seq_lst), 'cases')
for e in range(epochs):
    print('='*10, 'EPOCH ', e, '='*10)
    
print('-'*10, 'training complete')
'''


# attempt to save model
model.save('./sco50wav_spkr.pth')   
print('model saved')

