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
#case_path = './LegalUISRNN/data/SCOTUS_Processed/*/*'
case_path = '/scratch/jt2565/SCOTUS_Processed/*/*'

case_path = glob.glob(os.path.dirname(case_path))

trn_seq_lst = []
trn_cluster_lst = []
test_seq_lst = []
test_cluster_lst = []

verbose = True
train_cases = 3
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
    if i>=5: #cap at 5 files
        break
    

if False:
    print(type(trn_seq_lst))
    print(type(trn_seq_lst[0]))
    print(type(trn_seq_lst[0][0]))
    print(type(trn_cluster_lst))
    print(type(trn_cluster_lst[0]))
    print(type(trn_cluster_lst[0][0]))


if verbose:
    print('train') 
    print('='*50)
    for c in range(len(trn_seq_lst)):
        print('case', c+1)
        print('emb shape:', len(trn_seq_lst[c]))
        print('label shape:', len(trn_cluster_lst[c]))
        print('-'*50)
        for u in range(len(trn_seq_lst[c])):
            print('utterance', u)
            print('>'*10)
            print(np.shape(trn_seq_lst[c][u]))
            print(np.shape(trn_cluster_lst[c][u]))
            if all(ele == trn_cluster_lst[c][u][0] for ele in trn_cluster_lst[c][u]):
                print('label type=', type(trn_cluster_lst[c][u][0]))
                print("good labels")
            else:
                print("bad labels")
            print('<'*10)
            if u>=2:
                break
    print('='*50)

if verbose:
    print('test') 
    print('='*50)
    for c in range(len(test_seq_lst)):
        print('case', c+1)
        print('emb shape:', np.shape(test_seq_lst[c]))
        print('label shape:', np.shape(test_cluster_lst[c]))
        print('-'*50)
        for u in range(np.shape(test_seq_lst[c])[0]):
            print('utterance', u)
            print('>'*10)
            print(np.shape(test_seq_lst[c][u]))
            print(np.shape(test_cluster_lst[c][u]))
            if all(ele == test_cluster_lst[c][u][0] for ele in test_cluster_lst[c][u]):
                print('label type=', type(test_cluster_lst[c][u][0]))
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
model_args.enable_cuda = False
model_args.rnn_depth = 2
model_args.rnn_hidden_size = 128
training_args.learning_rate = 0.01
training_args.train_iteration = 200
training_args.enforce_cluster_id_uniqueness=False #based on dvec_SCOTUS
training_args.batch_size = 2
inference_args.test_iteration = 2 
model = uisrnn.UISRNN(model_args)

print('-'*10, 'training')
for c in range(len(trn_seq_lst)):
    train_sequences = trn_seq_lst[c]
    train_cluster_ids = trn_cluster_lst[c]
    if verbose:
        print('training case', c)
        print('list?', isinstance(train_sequences, list))
        print('item type?', type(train_sequences[0]))
        print('num utt', len(train_sequences))

    model.fit(train_sequences, train_cluster_ids, training_args)
if verbose:
    print('-'*10, 'training complete')


# attempt to save model
model.save('./localsamp_uisrnn.pth')  
print('model saved')


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


