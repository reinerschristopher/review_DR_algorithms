import numpy as np
import pandas as pd 
from sklearn.datasets import fetch_openml
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml

from sklearn.decomposition import PCA
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import os
import sys
import time
from ast import literal_eval as leval
import warnings

import argparse

from profilehooks import profile 

# python3 -m cProfile -o profile_file dimred.py

HOME_PATH = os.path.join(os.path.expanduser('~'))
LOCAL_DIR = os.path.join(HOME_PATH,'..', '..', 'local', 'hdd', 'reiners')
EXP_FILE = os.path.join(HOME_PATH, 'experiments.txt')
REP_PATH = os.path.join(HOME_PATH, 'rep')

# add openml data_ids
openml_dict = {'mnist':'554', 'fmnist':'40996'}

# add algorithm
alg_list = ['umap', 'tsne', 'opentsne', 'trimap', 'gumap', 'pca', 'gtsne']

# add source
src_list = ['cartoon', 'cartoon10k', 'cartoonpca748', 'cartoonpca5000']
[src_list.append(i) for i in list(openml_dict.keys())]


# run experiment 34 35 36 on mnist and fmnist dataset
#   python3 dimred -e 34 -r 3 -s mnist fmnist

parser = argparse.ArgumentParser()

# input source
parser.add_argument('-s', default=['cartoon'], nargs='*', required=False, type=str, 
                    help='choose {}'.format(src_list))

# input experiment number
parser.add_argument('-e', nargs='+', required=True, type=int, 
                    help='enter number of experiment as used in experiments.txt')

# input number of runs (number of consecutive experiments)
parser.add_argument('-r', default=1, required=False, type=int, 
                    help='starts -r consecutive experiments starting at -e')

# input number of samples
parser.add_argument('-n', default=-1, required=False, type=int, 
                    help='if no input use maximum number of samples')

# init with pca
parser.add_argument('--init', default=0, required=False, type=int, 
                    help='if no input use maximum number of samples')

args = parser.parse_args()

sources = args.s
n_samples = args.n
if args.init < 0:
  raise ValueError('init must be => 2')
else:
  init = args.init

if len(args.e) > 1:
  if args.r > 1:
    raise ValueError(' "-r" may just be supplied for one experiment.')
  else:
    experiments = args.e
  
else:
  experiments = range(args.e[0], args.e[0] + args.r)


for source in sources:
  # import source
  if source == 'cartoon':
    if n_samples == -1:
      n = 1
    elif n_samples < 11:
      n = n_samples
    else:
      n = int(n_samples/10000)
      if n > 10 or n % 10000 != 0:
        raise ValueError('Enter valid number of samples for cartoon.')

    array_list = []

    for i in range(n):
      SOURCE_FILE = os.path.join(LOCAL_DIR, 'cartoon', 'cartoon100k_' + str(i) + 'crop.npy')
      array = np.load(SOURCE_FILE)
      array = array.reshape((array.shape[0],-1))
      array_list.append(array)

    X = np.vstack([arr for arr in array_list])

  elif source == 'cartoon10k':
    if n_samples != -1:
      raise ValueError('cartoon10k just runs with 10k samples')

    SOURCE_FILE = os.path.join(LOCAL_DIR, 'cartoon10k_full.npy')
    X = np.load(SOURCE_FILE)
    X = X.reshape((X.shape[0],-1))

  elif source[:10] == 'cartoonpca':
    SOURCE_FILE = os.path.join(LOCAL_DIR, 'cartoonpca_'+source[10:]+'.npy')
    X = np.load(SOURCE_FILE)
    X = X.reshape((X.shape[0],-1))

  # get data from openml
  elif source in openml_dict.keys():
    SOURCE_FILE = os.path.join(HOME_PATH, 'openml_data')
    data_id = openml_dict[source]

    dataset = fetch_openml(data_id=data_id, data_home=SOURCE_FILE)

    if n_samples == -1:
      X = dataset.data
    elif n_samples < dataset.data.shape[0]:
      X = dataset.data[:n_samples]
    else:
      raise ValueError('Max. number of samples for {} is {}'.format(source, dataset.data.shape[0]))

  else:
    raise ValueError('Enter valid source {}'.format(src_list))

  print('Imported {} of shape ({},{})'.format(source, *X.shape))

  if init:
    pca_embedding = PCA(n_components=init)

    t = time.perf_counter()
    X = pca_embedding.fit_transform(X)
    pca_time = time.perf_counter() - t

    print('run PCA in {}'.format(pca_time))
    print('preprocessed data with PCA (reduced to {} dimensions, \
            explained_variance_ratio={})' \
            .format(init, np.sum(pca_embedding.explained_variance_ratio_)))

    if source == 'cartoon':
      PCA_FILE = os.path.join(LOCAL_DIR, 'cartoonpca' + '_' + str(init) + '.npy')
      np.save(PCA_FILE, X)

  for exp in experiments:
    # import model settings
    settings = {}

    if exp != -1:
      with open(EXP_FILE) as f:
        fl = f.readlines()

        for i in range(len(fl)):
          l = fl[i]
          if l == ('{}:\n'.format(str(exp))):
            algorithm = fl[i+1][:-1]
            if algorithm not in alg_list:
              raise ValueError('Please enter valid algorithm for experiment {}'.format(exp))
            settings = leval(fl[i+2])
            if type(settings) != dict:
              raise ValueError('Please enter valid dict for experiment {}'.format(exp))

    if len(settings) == 0:
      print('continue using default model settings (adds random_state=2364)')


    # initialize model
    if algorithm == 'umap':
      from umap import UMAP
      settings['random_state'] = 2364
      embedding = UMAP(**settings)

    elif algorithm == 'tsne':
      from sklearn.manifold import TSNE
      settings['random_state'] = 2364
      embedding = TSNE(**settings)

    elif algorithm == 'opentsne':
      from openTSNE import TSNE
      # CAUTION: openTSNE uses fit instead of fit_transform
      settings['random_state'] = 2364
      embedding = TSNE(**settings)

      t = time.perf_counter()
      rep = embedding.fit(X)
      fit_time = time.perf_counter() - t

    elif algorithm == 'trimap':
      from trimap import TRIMAP
      embedding = TRIMAP(**settings)

    elif algorithm == 'pca':
      embedding = PCA(**settings)

    elif algorithm == 'gtsne':
      from tsnecuda import TSNE 
      embedding = TSNE(**settings)

    elif algorithm == 'gumap':
      import cudf

      pdf = pd.DataFrame(X)
      _X = cudf.from_pandas(pdf)

      from cuml import UMAP
      embedding = UMAP(**settings)

      t = time.perf_counter()
      df_rep = embedding.fit_transform(_X)
      fit_time = time.perf_counter() - t

      rep = df_rep.as_matrix()

    else:
      raise ValueError('Please enter valid algorithm {}'.format(alg_list))

    if algorithm not in ['gumap', 'opentsne']:
      t = time.perf_counter()
      #@profile(filename=s.path.join(HOME_PATH, 'profile_'+algorithm+'_'+source))
      rep = embedding.fit_transform(X)
      fit_time = time.perf_counter() - t

    print('{} finished in {} minutes'.format(algorithm, fit_time/60))

    # Store result
    if init:
      REP_FILE = os.path.join(REP_PATH, source + '_' + str(exp) + '_pca_rep.npy')
    else:  
      REP_FILE = os.path.join(REP_PATH, source + '_' + str(exp) + '_rep.npy')
    np.save(REP_FILE, rep)


    # Add logging information
    if init:
      LOG_FILE = os.path.join(REP_PATH, source + '_' + str(exp) + '_pca_log.txt')
    else:
      LOG_FILE = os.path.join(REP_PATH, source + '_' + str(exp) + '_log.txt')  
    
    f = open(LOG_FILE, "a+")
    f.write('%============\n')
    f.write('Source:\n{}\n'.format(source))
    f.write('Algorithm:\n{}\n'.format(algorithm))
    f.write('Experiement:\n{}\n'.format(exp))
    f.write('n_samples, n_features:\n{}, {}\n'.format(*X.shape))
    f.write('fit time (perf_counter):\n{}\n'.format(fit_time))
    f.write('parameters:\n{}\n'.format(settings))
    if algorithm == 'pca':
      f.write('explained_variance_ratio:\n{}\n'.format(np.sum(embedding.explained_variance_ratio_)))
    f.write('%============\n')
    f.close()

  