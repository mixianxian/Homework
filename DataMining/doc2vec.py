import pandas as pd
import numpy as np
import random
import jieba
from smart_open import open
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

# Read csv file as numpy array
def csv_read(path):
	data = pd.read_csv(path)
	tag = data.loc[:,'id']
	text = data.loc[:,'text']
	label = data.loc[:,'label']
	return np.array(tag),np.array(text),np.array(label)

# Split train.csv into training.csv and testing.csv
# Save train_label.txt and test_label.txt

# training : testing = 34997:3471 = 10:1
def split(text,label):
	index = np.random.permutation(38471)
	test_index = index[:3471]

	test_text = text[test_index]
	test_label = label[test_index]

	train_text = np.delete(text,test_index,0)
	train_label = np.delete(label,test_index,0)

	'''
	train_label_data = pd.DataFrame(train_label,columns=['label'])
	train_label_data.to_csv('data/train_label.csv',index=False)
	test_label_data = pd.DataFrame(test_label,columns=['label'])
	test_label_data.to_csv('data/test_label.csv',index=False)
	'''

	#np.savetxt('data/train_label.txt',train_label,'%d')
	#np.savetxt('data/test_label.txt',test_label,'%d')

	return train_text,test_text,train_label,test_label

def segmentation(doc,label,savepath,stop_listpath=None):
	if stop_listpath != None:
		stop_list = [line[:-1] for line in open(stop_listpath,encoding='utf-8')]
	else:
		stop_list = []
	seg_doc = []
	for i,sentence in enumerate(doc):
		seg_cut = [word for word in jieba.cut(sentence) if word not in stop_list]
		if seg_cut == []:
			label = np.delete(label,i,0)
		else:
			seg_doc.append(' '.join(seg_cut))			
	print(len(seg_doc))
	seg_doc_data = pd.DataFrame(seg_doc,columns=['text'])
	seg_doc_data.to_csv(savepath,index=False)
	np.savetxt(savepath[:-7]+'label.txt',label,'%d')
	return seg_doc

# Try to use iterator to save memory
# which is probably unnecessary in this case
class TrainDoc:
	"""docstring for TrainDoc"""
	def __init__(self, filepath):
		self.filepath = filepath

	def __iter__(self):
		data = pd.read_csv(self.filepath)
		for i,line in enumerate(data.loc[:,'text']):
			yield TaggedDocument(line.split(),[i])
'''
class PredicDoc:
	"""docstring for PredicDoc"""
	def __init__(self, filepath):
		self.filepath = filepath

	def __iter__(self):
		data = pd.read_csv(self.filepath)
		for line in data.loc[:,'text']:
			yield line
'''
def train(modelpath,filepath,vector_size=50,window=3,min_count=20,epochs=10):
	train_text = TrainDoc(filepath)
	model = Doc2Vec(vector_size=vector_size,window=window,min_count=min_count,epochs=epochs,sample=1e-5,workers=4)
	model.build_vocab(train_text)
	model.train(train_text,total_examples=model.corpus_count, epochs=model.epochs)

	# save the trained model
	modelname = get_tmpfile(modelpath)
	model.save(modelname)	

def predict(modelpath,filepath,savepath):
	data = pd.read_csv(filepath)
	predict_text = data.loc[:,'text']

	model = Doc2Vec.load(modelpath)

	# Infer and save doc_vector of test docs
	doc_size = len(predict_text)
	vector_size = len(model.docvecs[0])
	doc_vec = np.zeros((doc_size,vector_size))
	for i in range(doc_size):
		if i%50 == 0:
			print(i)
		model.random.seed(0)
		doc_vec[i] = model.infer_vector(predict_text[i])
	np.savetxt(savepath,doc_vec,'%.12f')

def main():
	# Delete words that contained in stopwords list
	# Optional: None,'data/stopwords_small.txt','data/stopwords_big.txt'
	stop_listpath = 'data/stopwords_big.txt'

	# It seems like that the model.save function of gensim doc2vec requires an absolute directory path
	# Define the model name as stopwordslist_docvecsize_windows__mincount_epochs_model
	modelpath = 'C:\\Users\\Lix\\Documents\\GitFile\\Homework\\DataMining\\models\\big_50_5_30_30_model'
	

	# Process the raw data file
	_,text,label = csv_read('data/raw_train.csv')
	train_doc,test_doc,train_label,test_label = split(text, label)
	segmentation(train_doc,train_label,'data/train_seg.csv',stop_listpath)
	segmentation(test_doc,test_label,'data/test_seg.csv',stop_listpath)

	# Train the doc2vec model
	train(modelpath,'data/train_seg.csv',vector_size=100,window=5,min_count=30,epochs=30)

	# Predict doc_vector of docs
	predict(modelpath, 'data/train_seg.csv','data/train_vec.txt')
	predict(modelpath, 'data/test_seg.csv','data/test_vec.txt')

	'''
	Now we have four files under ./data/ directory:
	train_vec.txt, train_label.txt;
	test_vec.txt, test_label.txt

	From this, we can use other classifier to classify and predict the reality of our news
	'''


if __name__ == '__main__':
	'''
	In our training task, just use main() function.
	If you need to obtain doc_vector from test_seg.csv file, use predict() function instead

	If we try to use k-fold cross validation method, change split() function into a loop
	'''
	main()
	#predict('models/big_50_5_30_30_model', 'data/test_seg.csv','data/testvec.txt')
	#predict('models/big_50_5_30_30_model', 'data/train_seg.csv','data/trainvec.txt')


