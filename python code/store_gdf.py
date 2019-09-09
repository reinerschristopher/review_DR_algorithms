import numpy as np
import pandas as pd
import os

HOME_PATH = os.path.join(os.path.expanduser('~'))

# load DataFrame
SOURCE_PATH = os.path.join(HOME_PATH, '..', '..', 'local', 'hdd', 'reiners', 'cartoon')
print('loading from {}'.format(SOURCE_PATH))

array_list = []
for num in range(10):
	DIR_NUM = str(num)
	SOURCE_FILE = os.path.join(SOURCE_PATH, 'cartoon100k_' + DIR_NUM + 'crop.npy')
	array = np.load(SOURCE_FILE)
	array = array[:,100:-100,100:-100,:]
	array = array.reshape((array.shape[0],-1))
	array_list.append(array)
	
	print('appended {} of shape ({},{})'.format(DIR_NUM, *array.shape))

X = np.vstack([arr for arr in array_list])

# transform X to pandas df 
header = [str(i) for i in range(360000)]
pdf = pd.DataFrame(data=X, columns=header)
print(pdf.shape)

STORE_FILE = os.path.join(HOME_PATH,'..', '..', 'local', 'hdd', 'reiners', 'cartoon100k_crop.parq')
print('storing file to {}'.format(STORE_FILE))
pdf.to_parquet(STORE_FILE)

"""
USING cuDF:
"""
"""
header = [str(i) for i in range(360000)]
pdf = pd.DataFrame(columns=header)
print(type(pdf.columns.values.tolist()[0]))

for num in range(1):
        DIR_NUM = str(num)
        SOURCE_FILE = os.path.join(SOURCE_PATH, 'cartoon100k_' + DIR_NUM + 'crop.npy')
        array = np.load(SOURCE_FILE)
        array = array[:,100:-100,100:-100,:]
        array = array.reshape((array.shape[0],-1))

        tmp = pd.DataFrame(array, columns=header)
        pdf = pdf.append(tmp)
        print(type(pdf.columns.values.tolist()[0]))
        print(type(pdf.columns.values.tolist()[1000]))

        print('appended {} of shape ({},{})'.format(DIR_NUM, *tmp.shape))

STORE_FILE = os.path.join(HOME_PATH,'..', '..', 'local', 'hdd', 'reiners', 'cartoon10k_crop.parq')
print('storing file to {}'.format(STORE_FILE))
pdf.to_parquet(STORE_FILE)
"""