import pandas as pd
import numpy as np
import jieba
import gensim

# Read csv file as numpy array
def csv_read(path):
	data = pd.read_csv(path)
	text = data.ix[:,'text']
	label = data.ix[:,'label']
	return np.array(text),np.array(label)

# Split train.csv into training.csv and testing.csv
# training : testing = 35000:3664 = 10:1
def split():
	pass

# word segmentation and save result
def jieba_fenci(doc):
	fenci = []
	with open('fenci.txt','w',encoding='utf-8') as f:
		for sentence in doc:
			seg_list = list(jieba.cut(sentence))
			f.write(str(item)[1:-1]+'\n')
			fenci.append(seg_list)
	return fenci

# Try to use iterator to save memory
# which is probably unnecessary in this case
class Mysentences:
	"""docstring for Mysentences"""
	def __init__(self, filepath):
		self.fielpath = filepath

	def __iter__(self):
		for line in open(filepath):
			yield line.split()

# Training word2vec model and save result
def word2vec():
	sentences = Mysentences('data/training.csv')
	for i in range(5):
		print(sentences)
	model = gensim.model.word2vec(sentences,min_count=10,size=200,workers=4)
	modle.save('data/mymodel')

	'''
	# Save word vector
	with open('WordVector.txt','w') as f:
		for word in model.wv.vocab:
			f.write(word+str(model.wv[word]))
	'''
		
def main():
	text,label = csv_read('data/train.csv')
	seg_text = jieba_fenci(text)	

if __name__ == '__main__':
