import numpy as np
from sklearn.datasets import fetch_openml

import pandas as pd 

from imageio import imwrite


source = 'satellite'

openml_dict = {'mnist':'554', 'fmnist':'40996', 'satellite':'40900'}

data_id = openml_dict[source]

dataset = fetch_openml(data_id=data_id, data_home='../mnistdata')

# add path and store labels

# label_dict = {'0': 'T-shirt', '1': 'Trouser', '2': 'Pullover', '3': 'Dress', '4': 'Coat', '5': 'Sandal', '6': 'Shirt', '7': 'Sneaker', '8': 'Bag', '9': 'Ankle boot'}

# def num_to_label(x):
# 	return label_dict[str(x)]

# y = dataset.target

# df = pd.DataFrame(data=y, columns=['labels'])

# df['path'] = ['../'+source+'/'+source+'_images/'+str(i)+'.jpg' for i in range(len(y))]

# if source == 'fmnist':
# 	df['labels'] = df['labels'].apply(num_to_label)

# print(df.head())

# df.to_feather(source+'_meta.feather')

# convert X to png
X = dataset.data
# X = 255 - X
# i = 0
# #imwrite(source+'_images/'+str(i)+'.jpg', X[i,:].reshape((28,28)))
# [imwrite('../'+source+'/'+source+'_images/'+str(i)+'.jpg', X[i,:].reshape((28,28))) for i in range(len(X))]
y = dataset.target

print(X.shape)
print(type(X))

print(y.shape)
print(y.shape)
