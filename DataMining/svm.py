import numpy as np 
from sklearn import svm

def data_loading():
	train_vec = np.loadtxt('data/train_vec.txt')
	train_label = np.loadtxt('data/train_label.txt')

	test_vec = np.loadtxt('data/test_vec.txt')
	test_label = np.loadtxt('data/test_label.txt')

	return train_label,train_vec,test_label,test_vec

def train(train_vec,train_label,test_vec,test_label):
	clf = svm.SVC().fit(train_vec,train_label)
	score = clf.score(test_vec,test_label)
	print('Mean Accuracy: {}'.format(score))

	clf.get_params()

def predict(svm_params,test_vec):
	clf = svm.SVC().set_params()
	predict_label = clf.predict(test_vec)
	

if __name__ == '__main__':
	train_label,train_vec,test_label,test_vec = data_loading()

	train(train_label,train_vec,test_label,test_vec)