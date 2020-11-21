import numpy as np
import uisrnn

'''
- check toy data shape
- import case (generate Z if necc)
'''

toy_train = np.load('data/toy_training_data.npz')
toy_test = np.load('data/toy_testing_data.npz', allow_pickle=True)

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
