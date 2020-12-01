
import sys
sys.path.append("./LegalUISRNN")
import numpy as np
import torch
import glob
import os
import uisrnn


#expects processed cases in data folder (take from Google Drive or PRINCE)
case_path = '/scratch/jt2565/sco50/sco50wav_proc_case/'      # prince path
spkr_path = '/scratch/jt2565/sco50/sco50wav_proc_spkr/*/*'      # prince path

spkr_path = glob.glob(os.path.dirname(spkr_path))

train_cases = 2 #3 train, 2 test
trn_seq_lst = []
trn_cluster_lst = []
test_seq_lst = []
test_cluster_lst = []

verbose = True

if verbose:
    print("WAV (SVE) d-vec testing")
    print("\n", "="*50, "\n Processing case-embedded d-vec")
        
#load 5 case-embedded dvecs (with directory holding raw files)
for i, case in enumerate(os.listdir(case_path)):
    if case[-7:] == 'seq.npy':
        case_id = case.split('/')[-1].split('_')[0]
        
        train_sequence = np.load(case)
        train_clus = np.load(case[:-7]+'id.npy')
        train_cluster_id = []
        
        
        #converts labels to int for inference/testing
        for j in range(np.shape(train_clus)[0]):
            if i <= train_cases:
                train_cluster_id.append(train_clus[j])
            else:
                train_cluster_id.append(list(map(int, train_clus[j])))
        train_cluster_id = np.concatenate(train_cluster_id, axis = 0)           
        if verbose:
            if i > train_cases:
                print("-- Stored as test case --")
            else:
                print("-- Stored as train case --")
            print('Processed case:', case_id)
            print('emb shape:', np.shape(train_sequence))
            print('label shape:', np.shape(train_clus))
            print('flat label:', np.shape(train_cluster_id))    
               
        #add to training or testing list (for multiple cases       
        if i <= train_cases:
            trn_seq_lst.append(train_sequence)
            trn_cluster_lst.append(train_cluster_id)
        else:
            test_seq_lst.append(train_sequence)
            test_cluster_lst.append(train_cluster_id) 
        if i>=4:
            print(" ---- By case-embedded d-vec importation complete---- ")
            print('train items:', len(trn_seq_lst))
            print('test items:', len(test_seq_lst))
            break

    
#load 5 spkr-embedded dvecs (where each case has own dir)

if verbose:
    print("\n\n\n\n", "="*50, "\n Processing spkr-embedded d-vec")
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
    if i>=4:
        print(" ---- By spkr-embedded d-vec importation complete---- ")
        print('train items:', len(trn_seq_lst))
        print('test items:', len(test_seq_lst))
        break

