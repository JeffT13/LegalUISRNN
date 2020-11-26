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
case_path = './scratch/jt2565/SCOTUS_Processed/*/*'
case_path = glob.glob(os.path.dirname(case_path))

total_cases = len(case_path)
train_case = total_cases//10*8
print("# of training:", train_case)
print("# total cases:" , total_case)

train_sequences = []
train_cluster_ids = []
test_sequences = []
test_cluster_ids = []

verbose = True
for i, case in enumerate(case_path):

    case_id = case.split('/')[-1][:-7]
    temp_seq = np.load(case+'/'+case_id+'_sequence.npy', allow_pickle=True)
    temp_lab = np.load(case+'/'+case_id+'_cluster_id.npy', allow_pickle=True)

    if verbose:
        print(case)
        print(case_id)
        print(np.shape(temp_seq))
        print(np.shape(temp_lab))

    if i < train_case:
        train_sequences.append(temp_seq)
        train_cluster_ids.append(temp_lab)
    else:
        test_sequences.append(temp_seq)
        test_cluster_ids.append(temp_lab)    


if verbose:
    print(len(train_sequences))
    print(len(test_sequences))
    
#Define UISRNN
model_args, training_args, inference_args = uisrnn.parse_arguments()
model_args.verbosity=3
model_args.observation_dim=256 #from hparam techincally
model_args.enable_cuda = True
model_args.rnn_depth = 2
model_args.rnn_hidden_size = 8
training_args.learning_rate = 0.01
training_args.train_iteration = 200
training_args.enforce_cluster_id_uniqueness=False #based on dvec_SCOTUS
training_args.batch_size = 2
inference_args.test_iteration = 2

#model training **Use loop?**
print('---------------', 'training')
model = uisrnn.UISRNN(model_args)
model.fit(train_sequences, train_cluster_ids, training_args)
torch.save(model.state_dict(), './scsamp_uisrnn.pth')
if verbose:
    print('---------------', 'training complete')

#evaluation has similar mechanic
predicted_cluster_ids = model.predict(test_sequences, inference_args)

model.logger.print(
        3, 'Asserting the equivalence between'
        '\nGround truth: {}\nPredicted: {}'.format(
            test_cluster_id, predicted_label))
    accuracy = uisrnn.compute_sequence_match_accuracy(
        predicted_label, test_cluster_id)
    self.assertEqual(1.0, accuracy)

