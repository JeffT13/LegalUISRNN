''' SCOTUS d-vec UISRNN processing'''
import sys
sys.path.append("./LegalUISRNN")
import numpy as np
import torch
import os
import uisrnn

case_path = '/scratch/jt2565/sco50/sco50wav_proc_case/'
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
            
            
            

#Define UISRNN (**copy from training**)
model_args, inference_args = uisrnn.parse_arguments()
model_args.verbosity=3 #can verbose=False for no prints except training
model_args.observation_dim=256 #from hparam
model_args.enable_cuda = True
model_args.rnn_depth = 2
#model_args.crp_alpha = .5
model_args.rnn_hidden_size = 32



model = uisrnn.UISRNN(model_args)

#load model
model.load('~/hold/sco50wav_case.pth')

#inference and evaluation
predicted_label = model.predict(test_seq_lst, inference_args)
model.logger.print(3, 'Asserting the equivalence between'
        '\nGround truth: {}\nPredicted: {}'.format(
            test_cluster_lst, predicted_label))
accuracy = uisrnn.compute_sequence_match_accuracy(predicted_label, test_cluster_lst)
print("--", accuracy, "--")


