import pandas as pd
import numpy as np
import jieba

# Read csv file as numpy array
def csv_read(path):
	data = pd.read_csv(path)
	text = data.ix[:,'text']
	label = data.ix[:,'label']
	return np.array(text),np.array(label)

# Split train.csv into training.csv and testing.csv
# Methods:
# training : testing = 
def split():
	pass

def jieba_fenci(doc):
	seg_list = [list(jieba.cut(sentence)) for sentence in doc[:100]]

	# To save it properly
	seg_save = [str(item)[1:-1]+'\n' for item in seg_list]
	with open('fenci.txt','w',encoding='utf-8') as f:
		f.writelines(seg_save)
	return seg_list

if __name__ == '__main__':
	text,label = csv_read('data\\train.csv')
	seg_list = jieba_fenci(text)