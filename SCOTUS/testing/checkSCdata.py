
import sys
sys.path.append("./LegalUISRNN")
import numpy as np
import torch
import glob
import os
import uisrnn


#expects processed cases in data folder (take from Google Drive or PRINCE)
#case_path = './LegalUISRNN/data/SCOTUS_Processed/*/*'  # local path
case_path = '/scratch/jt2565/SCOTUS_Processed/*/*'      # prince path


case_path = glob.glob(os.path.dirname(case_path))
total_cases = len(case_path)
train_cases = total_cases//10*8
print("# of training:", train_cases)
print("# total cases:" , total_cases)

trn_seq_lst = []
trn_cluster_lst = []
test_seq_lst = []
test_cluster_lst = []

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
               
    if False:
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



from uisrnn import utils
item=0
(concatenated_train_sequence, concatenated_train_cluster_id) = utils.concatenate_training_data(trn_seq_lst[item], trn_cluster_lst[item], False, False)

if verbose:
    print(type(concatenated_train_sequence), type(concatenated_train_sequence[0]))
    #print(np.shape(concatenated_train_sequence))
    print(np.shape(concatenated_train_sequence))

