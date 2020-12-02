import sys
sys.path.append("./LegalUISRNN")
import numpy as np
import torch
import glob
import os, csv
import uisrnn


#expects processed cases in data folder
case_path = '/scratch/jt2565/sco50/sco50wav_proc_case/' 
model_path = './hold/sco50wav_case.pt'

#case_path = '/mnt/c/Fall2020/Capstone/LegalUISRNN/data/sco50wav_proc_case/'
#model_path = '/mnt/c/Fall2020/Capstone/LegalUISRNN/data/sco50wav_case.pth'


total_cases = (len(os.listdir(case_path))/2)
train_cases = (total_cases//10)*9
print("# of training:", train_cases)
print("# total cases:" , total_cases)
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
        
        train_sequence = np.load(case_path+case)
        train_clus = np.load(case_path+case_id+'_id.npy')
        train_cluster_id = []
        
        
        #converts labels to int for inference/testing
        for j in range(np.shape(train_clus)[0]):
            if i <= train_cases:
                train_cluster_id.append(str(train_clus[j]))
            else:
                train_cluster_id.append(int(train_clus[j]))
            if j==(np.shape(train_clus)[0]-1):        
                train_cluster_id = np.asarray(train_cluster_id)
                
        if verbose:
            if i > train_cases:
                print("-- Stored as test case --")
            else:
                print("-- Stored as train case --")
            print('Processed case:', case_id)
            print('TYPES(seq, id):', type(train_sequence), type(train_cluster_id))
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
        if False: # i>=4 careful with enumerate as files get skipped by i iterates
            print(" ---- By case-embedded d-vec importation complete---- ")
            print('train items:', len(trn_seq_lst))
            print('test items:', len(test_seq_lst))
            break


with open('./predicted_labels.csv', newline='') as f:
    reader = csv.reader(f)
    pred = list(reader)

ans = test_cluster_lst[0]

if verbose:
    print("-- Inference --")
    print(type(pred))
    print(type(ans))
    print(len(pred), len(ans))
    
uisrnn.compute_sequence_match_accuracy(pred, ans)
print("--", accuracy, "--")

'''
# Visualize dvecs
concat_trn = np.concatenate(trn_seq_lst)
print(concat_trn.shape)
print(np.max(concat_trn))
print(np.min(concat_trn))
print(np.isnan(concat_trn).any()) 
'''
