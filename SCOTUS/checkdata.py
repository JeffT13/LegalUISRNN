import numpy as np
import sys
sys.path.append("./LegalUISRNN")
import uisrnn

'''
- check toy data shape
- import case (generate Z if necc)
'''

toy_train = np.load('./LegalUISRNN/data/toy_training_data.npz')
toy_test = np.load('./LegalUISRNN/data/toy_testing_data.npz', allow_pickle=True)

print('Train:', toy_train.files)
print(np.shape(toy_train['train_sequence']))
print(np.shape(toy_train['train_cluster_id']))
print(np.unique(toy_train['train_cluster_id']))
print(np.shape(np.unique(toy_train['train_cluster_id'])))

print(type(toy_train['train_cluster_id'][0]))
print('Test:', toy_test.files)
print(np.shape(toy_test['test_sequences']))
print(np.shape(toy_test['test_cluster_ids']))
print(np.shape(np.unique(toy_test['test_cluster_ids'])))
print(np.shape(toy_test['test_sequences'][0]))
print(np.unique(toy_test['test_cluster_ids'][0]))
print(np.shape(np.unique(toy_test['test_cluster_ids'][0])))
print(np.shape(toy_test['test_sequences'][1]))
print(np.unique(toy_test['test_cluster_ids'][1]))
print(np.shape(np.unique(toy_test['test_cluster_ids'][1])))


"""
training is one long sequence
testing is multiple isolated examples

we might do training in examples as well (by case)
	- make sure longest case fits in GPU****

"""

our_seq = np.load('./LegalUISRNN/data/SCOTUS_Processed/17-1268_SCOTUS/17-1268_sequence.npy', allow_pickle=True)
our_id = np.load('./LegalUISRNN/data/SCOTUS_Processed/17-1268_SCOTUS/17-1268_cluster_id.npy', allow_pickle=True)

print('SCOTUS 17-1268')
print(np.shape(our_seq))
print(np.shape(our_id))
print(np.unique(our_id))
print(np.shape(np.unique(our_id)))