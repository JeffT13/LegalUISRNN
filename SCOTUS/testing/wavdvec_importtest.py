
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
    print('WAV (SVE) d-vec testing')
    
    
#load 5 case-embedded dvecs (with directory holding raw files)
for i, case in enumerate(os.listdir(case_path)):
    if case[-7:] == 'seq.wav':
        case_id = case.split('/')[-1].split('_')[0]
        
        
        train_sequence = np.load(case+'/'+case_id+'_seq.npy')
        train_clus = np.load(case+'/'+case_id+'_id.npy')
        train_cluster_id = []
        
        
        #converts labels to int for inference/testing
        for j in range(np.shape(train_clus)[0]):
            if i <= train_cases:
                train_cluster_id.append(train_clus[j])
            else:
                train_cluster_id.append(list(map(int, train_clus[j])))
                   
        if verbose:
            
            if i > train_cases:
                    print("-- Stored as test case --")
            print('Processed case:', case_id)
            print('emb shape:', np.shape(train_sequence))
            print('label shape:', np.shape(train_clus))
            print('emb len:', len(train_sequence))
            print('label len:', len(train_cluster_id))    
               
        #add to training or testing list (for multiple cases       
        if i <= train_cases:
            trn_seq_lst.append(train_sequence)
            trn_cluster_lst.append(train_cluster_id)
        else:
            test_seq_lst.append(train_sequence)
            test_cluster_lst.append(train_cluster_id) 
        if i>=4:
            print(" ---- By case-embedded d-vec importation complete---- ")
            break

    
#load 5 spkr-embedded dvecs (where each case has own dir)
for i, casespkr in enumerate(spkr_path):

    case_id = casespkr.split('/')[-1][:-7]
    train_sequence = np.load(casespkr+'/'+case_id+'_sequence.npy')
    train_clus = np.load(casespkr+'/'+case_id+'_cluster_id.npy')
    train_cluster_id = []
    
    #converts labels to int for inference/testing
    for j in range(np.shape(train_clus)[0]):
        if i <= train_cases:
            train_cluster_id.append(train_clus[j])
        else:
            train_cluster_id.append(list(map(int, train_clus[j])))
               
    if verbose:
        
        if i > train_cases:
                print("-- Stored as test case --")
        print('Processed case:', case_id)
        print('emb shape:', np.shape(train_sequence))
        print('label shape:', np.shape(train_clus))
        print('emb len:', len(train_sequence))
        print('label len:', len(train_cluster_id))    
           
    #add to training or testing list (for multiple cases       
    if i <= train_cases:
        trn_seq_lst.append(train_sequence)
        trn_cluster_lst.append(train_cluster_id)
    else:
        test_seq_lst.append(train_sequence)
        test_cluster_lst.append(train_cluster_id) 
    if i>=4:
        print(" ---- By spkr-embedded d-vec importation complete---- ")
        break

