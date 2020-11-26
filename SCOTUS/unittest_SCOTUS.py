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
case_path = '/scratch/jt2565/SCOTUS_Processed/*/*'
case_path = glob.glob(os.path.dirname(case_path))

trn_seq_lst = []
trn_cluster_lst = []
test_seq_lst = []
test_cluster_lst = []

verbose = True
train_cases = 4
for i, case in enumerate(case_path):

    case_id = case.split('/')[-1][:-7]
    train_sequence = np.load(case+'/'+case_id+'_sequence.npy', allow_pickle=True)
    train_cluster_id = np.load(case+'/'+case_id+'_cluster_id.npy', allow_pickle=True)

    if False:
        print(case_id)
        print('emb shape:', np.shape(train_sequence))
        print('label shape:', np.shape(train_cluster_id))

    if i <= train_cases:
        trn_seq_lst.append(train_sequence)
        trn_cluster_lst.append(train_cluster_id)
    else:
        test_seq_lst.append(train_sequence)
        #convert label strings to int
        for i in range(np.shape(train_cluster_id)[0]):
            train_cluster_id[i] = list(map(int, train_cluster_id[i]))
        test_cluster_lst.append(train_cluster_id) 


if verbose:
    print('train') 
    print('='*50)
    for c in range(len(trn_seq_lst)):
        print('case', c+1)
        print('emb shape:', np.shape(trn_seq_lst[c]))
        print('label shape:', np.shape(trn_cluster_lst[c]))
        print('-'*50)
        for u in range(np.shape(trn_seq_lst[c])[0]):
            print('utterance', u)
            print('>'*10)
            print(np.shape(trn_seq_lst[c][u]))
            print(np.shape(trn_cluster_lst[c][u]))
            #print('+'*5)
            #print(trn_seq_lst[c][u][:3])
            #print(trn_cluster_lst[c][u][0:3])
            if all(ele == trn_cluster_lst[c][u][0] for ele in trn_cluster_lst[c][u]):
                print("good labels")
            else:
                print("bad labels")
            print('<'*10)
            if u>=2:
                break
    print('='*50)
 

#Define UISRNN
model_args, training_args, inference_args = uisrnn.parse_arguments()
model_args.verbosity=3
model_args.observation_dim=256 #from hparam
model_args.enable_cuda = True
model_args.rnn_depth = 2
model_args.rnn_hidden_size = 8
training_args.learning_rate = 0.01
training_args.train_iteration = 200
training_args.enforce_cluster_id_uniqueness=False #based on dvec_SCOTUS
training_args.batch_size = 2
inference_args.test_iteration = 2 


print('-'*10, 'training')
model = uisrnn.UISRNN(model_args)

for c in range(len(trn_seq_lst)):
    train_sequences = trn_seq_lst[c]
    train_cluster_ids = trn_cluster_lst[c]
    if verbose:
        print('training case', c)
       
    if True:
        train_sequences.to('cuda')
        train_cluster_ids.to('cuda')
    model.fit(train_sequences, train_cluster_ids, training_args)
if verbose:
    print('-'*10, 'training complete')

# attempt to save model
model.save('./localsamp_uisrnn.pth')  
print('model saved')
if True:
    print('-'*10, 'testing')
    for c in range(len(trn_seq_lst)):
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
        break  


        
if False:       
    model.logger.print(3, 'Asserting the equivalence between \nGround truth: {}\nPredicted: {}'.format(test_cluster_id, predicted_label))
    print('Asserting the equivalence between','\nGround truth: {}\nPredicted: {}'.format(test_cluster_id, predicted_label))
    accuracy = uisrnn.compute_sequence_match_accuracy(predicted_label, test_cluster_id)
    print('acc:', accuracy)

    model.save('./scsamp_uisrnn.pth')
