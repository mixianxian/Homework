import pandas as pd
import numpy as np
import jieba
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
# Methods:
# training : testing = 
def split():
	pass

def segmentation(doc,path=None):
	if path != None:
		stop_list = [line[:-1] for line in open(path)]
	else:
		stop_list = []
	seg_doc = []
	with open('data/seg_words.txt','w',encoding='utf-8') as f:
		for sentence in doc:
			seg_result = [word for word in jieba.cut(sentence) if word not in stop_list]
			seg_doc.append(seg_result)
			f.write(' '.join(seg_result)+'\n')
	return seg_doc

class TrainDoc:
	"""docstring for TrainDoc"""
	def __init__(self, filepath):
		self.filepath = filepath

	def __iter__(self):
		for i,line in enumerate(open(self.filepath)):
			yield TaggedDocument(line[:-1].split(),[i])

class PredicDoc:
	"""docstring for PredicDoc"""
	def __init__(self, filepath):
		self.filepath = filepath

	def __iter__(self):
		for line in open(filepath):
			yield line[:-1].split()

def train(modelpath,filepath,vector_size=50,window=3,min_count=10,epochs=10):
	train_text = TrainDoc(filepath)
	model = Doc2Vec(vector_size=vector_size,window=window,min_count=min_count,epochs=epochs,sample=1e-5,workers=4)
	model.build_vocab(train_text)
	model.train(train_text,total_examples=model.corpus_count, epochs=model.epochs)

	modelname = get_tmpfile(modelpath)
	model.save(modelname)

	# save these trained doc vector
	l = len(list(train_text))
	doc_vec = np.zeros((l,vector_size))
	for i in range(l):
		doc_vec[i] = model.docvecs[i]
	np.savetxt('data/train_vec.txt',doc_vec)
	

def predict(modelpath,filepath):
	predict_text = PredicDoc(filepath)
	model = Doc2Vec.load(modelpath)

if __name__ == '__main__':
	_,text,label = csv_read('data/train.csv')
	seg_doc = segmentation(text[:100])
	#train('C:\\Users\\xiang\\Documents\\GitFile\\Homework\\DataMining\\models\\modelstest_model','data/seg_words.txt')
	model = Doc2Vec.load('C:\\Users\\xiang\\Documents\\GitFile\\Homework\\DataMining\\models\\modelstest_model')
	print(model.infer_vector(seg_doc[52]))