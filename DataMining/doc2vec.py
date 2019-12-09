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
	tag = data.ix[:,'id']
	text = data.ix[:,'text']
	label = data.ix[:,'label']
	return np.array(tag),np.array(text),np.array(label)

# Split train.csv into training.csv and testing.csv
# Save train_label.txt and test_label.txt

# training : testing = 35000:3471 = 10:1
def split(text,label):
	index = np.arange(38471)
	random.shuffle(index)
	test_index = index[:3471]

	test_text = text[test_index]
	test_label = label[test_index]

	train_text = np.delete(text,test_index,0)
	train_label = np.delete(label,test_index,0)

	np.savetxt('data/train_label.txt',train_label,'%d')
	np.savetxt('data/test_label.txt',test_label,'%d')

	return train_text,test_text

def segmentation(doc,savepath,stop_listpath=None):
	if stop_listpath != None:
		stop_list = [line[:-1] for line in open(stop_listpath,encoding='utf-8')]
	else:
		stop_list = []
	seg_doc = []
	with open(savepath,'w',encoding='utf-8') as f:
		for sentence in doc:
			seg_result = [word for word in jieba.cut(sentence) if word not in stop_list]
			seg_doc.append(seg_result)
			f.write(' '.join(seg_result)+'\n')
	return seg_doc

# Try to use iterator to save memory
# which is probably unnecessary in this case
class TrainDoc:
	"""docstring for TrainDoc"""
	def __init__(self, filepath):
		self.filepath = filepath

	def __iter__(self):
		for i,line in enumerate(open(self.filepath,encoding='utf-8')):
			yield TaggedDocument(line[:-1].split(),[i])

class PredicDoc:
	"""docstring for PredicDoc"""
	def __init__(self, filepath):
		self.filepath = filepath

	def __iter__(self):
		for line in open(self.filepath,encoding='utf-8'):
			yield line[:-1].split()

def train(modelpath,filepath,vector_size=50,window=3,min_count=20,epochs=10):
	train_text = TrainDoc(filepath)
	model = Doc2Vec(vector_size=vector_size,window=window,min_count=min_count,epochs=epochs,sample=1e-5,workers=4)
	model.build_vocab(train_text)
	model.train(train_text,total_examples=model.corpus_count, epochs=model.epochs)

	# save the trained model
	modelname = get_tmpfile(modelpath)
	model.save(modelname)

	# save these trained doc vector
	doc_size = len(list(train_text))
	doc_vec = np.zeros((doc_size,vector_size))
	for i in range(doc_size):
		doc_vec[i] = model.docvecs[i]
	np.savetxt('data/train_vec.txt',doc_vec,'%.12f')
	

def predict(modelpath,filepath):
	predict_text = PredicDoc(filepath)
	model = Doc2Vec.load(modelpath)

	# Infer and save doc_vector of test docs
	doc_size = len(list(predict_text))
	vector_size = len(model.docvecs[0])
	doc_vec = np.zeros((doc_size,vector_size))
	for i in range(doc_size):
		doc_vec[i] = model.docvecs[i]
	np.savetxt('data/test_vec.txt',doc_vec,'%.12f')

def predict_fromfile(modelpath,filepath):
	stop_listpath = 'data/stopwords_big.txt'
	modelpath = 'C:\\Users\\Lix\\Documents\\GitFile\\Homework\\DataMining\\models\\big_50_3_10_10_model'

	data = pd.read_csv(filepath)
	text_doc = data.ix[:,'text']
	segmentation(test_doc,'data/test_seg.txt',stop_listpath)
	predict(modelpath, 'data/test_seg.txt')


def main():
	# Delete words that contained in stopwords list
	# Optional: None,'data/stopwords_small.txt','data/stopwords_big.txt'
	stop_listpath = 'data/stopwords_big.txt'

	# It seems like that the model.save function of gensim doc2vec requires an absolute directory path
	# Define the model name as stopwordslist_docvecsize_windows__mincount_epochs_model
	modelpath = 'C:\\Users\\Lix\\Documents\\GitFile\\Homework\\DataMining\\models\\big_50_3_20_10_model'

	# Process the raw data file
	_,text,label = csv_read('data/train.csv')
	train_doc,test_doc = split(text, label)
	segmentation(train_doc,'data/train_seg.txt',stop_listpath)
	segmentation(test_doc,'data/test_seg.txt',stop_listpath)

	# Train the doc2vec model
	train(modelpath,'data/train_seg.txt',vector_size=50,window=3,min_count=10,epochs=10)

	# Predict doc_vector of test docs
	predict(modelpath, 'data/test_seg.txt')

	'''
	Now we have four files under ./data/ directory:
	train_seg.txt, train_label.txt;
	test_seg.txt, test_label.txt

	From this, we can use other classifier to classify and predict the reality of our news
	'''


if __name__ == '__main__':
	'''
	In our training task, just use main() function.
	If you need to obtain doc_vector from test.csv file, use predict_fromfile() function instead

	If we try to use k-fold cross validation method, change split() function into a loop
	'''
	main()