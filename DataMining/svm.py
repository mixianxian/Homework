import numpy as np 
from sklearn import svm
import pickle

def data_loading():
	train_vec = np.loadtxt('data/train_vec.txt')
	train_label = np.loadtxt('data/train_label.txt')

	test_vec = np.loadtxt('data/test_vec.txt')
	test_label = np.loadtxt('data/test_label.txt')

	return train_label,train_vec,test_label,test_vec

def train(train_vec,train_label,model_path):
	clf = svm.SVC(gamma='auto')
	clf.fit(train_vec,train_label)

	with open(model_path,'wb') as f:
		pickle.dump(clf,f)

def predict(model_path,test_vec,test_label):
	with open(model_path,'rb') as f:
		clf = pickle.load(f)
		score = clf.score(test_vec,test_label)
		print('Mean Accuracy: {}'.format(score))
		#predict_label = clf.predict(test_vec)

def main(model_path):
	train_label,train_vec,test_label,test_vec = data_loading()
	train(train_vec,train_label,model_path)
	predict(model_path,test_vec,test_label)	


if __name__ == '__main__':
	model_path = 'models/params_model.svm'

	main(model_path)

