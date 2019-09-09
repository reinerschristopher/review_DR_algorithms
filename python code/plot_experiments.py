# 0  %============
# 1  Source:
# 2  mnist
# 3  Algorithm:
# 4  umap
# 5  Experiement:
# 6  70
# 7  n_samples, n_features:
# 8  70000, 784
# 9  fit time (perf_counter):
# 10 224.53329402394593
# 11 parameters:
# 12 {'n_neighbors': 5, 'random_state': 2364}
# 13 %============

# 0  %============
# 1  Source:
# 2  mnist
# 3  Algorithm:
# 4  umap
# 5  Experiement:
# 6  70
# 7  n_samples, n_features:
# 8  70000, 784
# 9  fit time (perf_counter):
# 10 224.53329402394593
# 11 parameters:
# 12 {'n_neighbors': 5, 'random_state': 2364}
# 13 %============

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
hv.renderer('bokeh').theme = 'light_minimal'  # more LaTeX looking style

import bokeh
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread
from holoviews.plotting.links import DataLink

from itertools import combinations  #used to DataLink all subplots
from ast import literal_eval as leval

import feather
import pickle

import os

#FILE_PATH = os.path.basename(os.path.dirname(__file__))
REP_PATH = os.path.join('..', 'data', 'rep')
CARTOON_PATH = os.path.join('..', 'cartoon')
META_PATH = os.path.join('..')


def load_cartoon_meta(n_file=1, src='cartoon'):
    with open(os.path.join(CARTOON_PATH, 'attr_dict.pickle'), 'rb') as handle:
        attr_dict = pickle.load(handle)

    if src == 'cartoon10k':
    	df = pd.read_feather(os.path.join(CARTOON_PATH, 'cartoon10k_meta.feather'))
        #raise ValueError('Import cartoon10k meta.')
    else: 
        df_list = []
        for i in range(n_file):
            df_list.append(pd.read_feather(os.path.join(CARTOON_PATH, 'meta100k', 'cartoon100k_'+ str(i) +'_meta.feather'))) 
        df = df_list[0]

        for i in range(1, n_file):
            df = df.append(df_list[i])

    return df, attr_dict


def add_experiments(experiments, src, df):
    for exp in experiments:
        if not any('x_'+str(exp) == s for s in df.columns.tolist()): 
            file_name = os.path.join(REP_PATH, src + '_rep', src + '_' + str(exp) + '_rep.npy')
            rep = np.load(file_name)

            n_samples = len(df)

            df['x_'+str(exp)] = rep[:n_samples,0]
            df['y_'+str(exp)] = rep[:n_samples,1]

            print('Experiment {} added.'.format(exp))
        #else:
            #warnings.warn('Experiment {} already in df.'.format(exp))

    if df.isna().sum().sum():
        warn('Please check that data is complete. There are NaNs.')

    return df


def get_label(exp, src):
    settings = None
    with open(os.path.join(REP_PATH, src + '_rep', src + '_' + str(exp) + '_log.txt'), 'r') as f:
        fl = f.readlines()
        algorithm = fl[4][:-1]
        shape = fl[8][:-1]
        settings = leval(fl[12])
        if 'random_state' in settings:
            del settings['random_state']

    return algorithm + ' trained on (' +shape + ') with: ' + str(settings)


def plot_experiments(src, experiments, attr=None, n_samples=10000, single_mode=False, width=550, n_cols=3, cmap=None):
    if type(experiments) == int:
        experiments = [experiments]

    if src == 'mnist' or src == 'fmnist':
        attr = 'labels'
        cmap = 'Category10'
        df = pd.read_feather(os.path.join(META_PATH, src, src+'_meta.feather'))
        df = df.loc[:n_samples]

    elif src == 'cartoon' or src == 'cartoon10k':
        n_file = int(np.ceil(n_samples/10000))
        df, attr_dict = load_cartoon_meta(n_file=n_file, src=src)
        df = df.loc[:n_samples]

        if attr is None:
            print('TODO: implement DynamicMap for src=cartoon and attr=None')
            attr = 'nhair'

        if cmap is None:
            if attr == 'nhair':
                cmap = 'Category20'
            else:
                cmap = 'viridis'


    df = add_experiments(experiments, src, df)

    if n_samples > 10000:
        scatter_options = opts.Scatter(width=width, aspect=1, xaxis=None, yaxis=None,
    	                               colorbar=False, cmap=cmap,
    	                               nonselection_alpha=0.01, nonselection_color='black')

        return hv.Layout([dynspread(datashade(hv.Scatter(df, 'x_'+str(e), ['y_'+str(e), attr], label=get_label(e, src)).opts(scatter_options), 
        									  aggregator=ds.count_cat(attr))) 
        				  for e in experiments])
    else: 
    	str_tooltips = "\
    	        <div> \
    	            <img src=\"@path\" width=\"100\" height=\"100\"></img> \
    	        </div> \
    	        <div> \
    	            <span style=\"font-size: 12px;\">label: @labels</span> \
    	        </div> \
    	        "
    	    
    	hover = bokeh.models.HoverTool(tooltips=str_tooltips)

    	scatter_options = opts.Scatter(width=width, aspect=1, xaxis=None, yaxis=None,
    	                               colorbar=False, cmap=cmap,
    	                               nonselection_alpha=0.01, nonselection_color='black',
    	                               tools=[hover,'box_select', 'lasso_select'])

    	if single_mode:
    	    raise ValueError('Implement single_mode (no layout just DynamicMap)')  # TODO: single_mode

    	scatter_list = [hv.Scatter(df, 'x_'+str(e), ['y_'+str(e), attr, 'path', 'labels'], label=get_label(e, src)).opts(scatter_options).opts(color=hv.dim(attr)) 
    	                for e in experiments]

    	if len(scatter_list) > 2:
    	    [DataLink(scatter1, scatter2) for scatter1, scatter2 in combinations(scatter_list, 2)]
    	    DataLink(scatter_list[0], scatter_list[1])

    	return hv.Layout(scatter_list).cols(n_cols).relabel(src + ' plotting ' + str(n_samples) + ' datapoints')


