#!/usr/bin/env python
# coding: utf-8

# ## Higgs Effective Field Theory (HEFT) Study: Training
# > Created: Feb 12, 2024 Nicola de Filippis, Kurtis Johnson, Harrison B. Prosper<br>
# > Updated: Mar 25, 2024 HBP
# 
# ### Introduction
# 
# The HEFT parameter space is defined by the 5 parameters $\theta = c_{hhh}, c_{t}, c_{tt}, c_{ggh}, c_{gghh}$. In this proof-of-principle, we set $c_{hhh} = c_{t} = 1$, which reduces the parameter space to 3 dimensions and yields the expression,
# 
# \begin{align}
#     \sigma(m_{gg}, \theta) & = \boldsymbol{c}^T(\theta) \cdot \boldsymbol{b}(m_{hh}), 
# \end{align}
# 
# for the cross section per bin, $\sigma$, where
# 
# \begin{align}
#     \boldsymbol{c}^T(\theta) & = (1, 
#                  c_{tt}, 
#              c_{ggh}, 
#              c_{gghh}, \nonumber\\
#              &\quad\,\,\,\, c_{tt} c_{ggh},
#              c_{tt} c_{gghh}, 
#              c_{ggh}c_{gghh}, \nonumber\\
#              &\quad\,\,\,\, c_{tt} c_{ggh}^2, 
#              c_{gghh} c_{ggh}^2, 
#              c_{tt}^2, 
#              c_{gghh}^2, 
#              c_{ggh}^2, 
#              c_{ggh}^3), 
# \end{align}
# 
# is a row matrix of polynomials in the HEFT parameters
# and $\boldsymbol{b}(m_{hh})$ is a column matrix of coefficients.
# 
# ### Model
# 
# In this notebook, we model the HEFT di-Higgs cross section[1] (per 15 GeV in the di-Higgs mass, $m_{hh}$) directly, that is, we model the function: $f: m_{hh}, c_{tt}, c_{ghh}, c_{gghh} \rightarrow \sigma$ directly using the training data prepared in the notebook `heft_prepare_traindata.ipynb`.
# 
# ### References
#   1. Lina Alasfar *et al.*, arXiv:2304.01968v1

# In[1]:


import os, sys

# the standard module for tabular data
import pandas as pd

# the standard module for array manipulation
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# to reload modules
import importlib

# some simple dnn untilities
import dnnutil as dn

# In[2]:

# set a seed to ensure reproducibility
seed = 42
rnd  = np.random.RandomState(seed)

# ### Load training data

# In[3]:


datafile = f'../data/heft_traindata.csv'

print('loading %s' % datafile)
df  = pd.read_csv(datafile)
print('number of rows: %d\n' % len(df))

df['target'] = df.sigma

print(f'min(sigma):  {df.target.min():10.3f} pb, '\
      f'avg(sigma):  {df.target.mean():10.3f} pb,  max(sigma): {df.target.max():10.3f} pb\n')


# ### Load $m_{hh}$ spectra

# In[4]:


spectra = pd.read_csv('../data/heft_spectra.csv')
print(len(spectra), spectra[:5])


# ### Train, validation, and test sets
# There is some confusion in terminology regarding validation and test samples (or sets). We shall adhere to the defintions given here https://machinelearningmastery.com/difference-test-validation-datasets/):
#    
#   * __Training Dataset__: The sample of data used to fit the model.
#   * __Validation Dataset__: The sample of data used to decide 1) whether the fit is reasonable (e.g., the model has not been overfitted), 2) decide which of several models is the best and 3) tune model hyperparameters.
#   * __Test Dataset__: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
# 
# The validation set will be some small fraction of the training set and can be used, for example, to decide when to stop the training.

# In[5]:


# Fraction of the data assigned as test data and validation
ntrain    = 20000                # training sample size
tfraction = (1-ntrain/len(df))/2 # test fraction
vfraction = tfraction            # validation fraction

# Split data into a part for training, validation, and testing
train_data, valid_data, test_data = dn.split_data(df, 
                                         test_fraction=tfraction, 
                                         validation_fraction=vfraction) 

print('train set size:        %6d' % train_data.shape[0])
print('validation set size:   %6d' % valid_data.shape[0])
print('test set size:         %6d' % test_data.shape[0])


# ### Empirical risk (that is, average loss)
# 
# The empirical risk, which is the **objective function** we shall minimize, is defined by
# 
# \begin{align}
# R_M(\omega) & = \frac{1}{M} \sum_{m=1}^{M} L(t_m, f_m),
# \end{align}
# 
# where 
# 
# \begin{align*}
#     f_m & \equiv f(\boldsymbol{x}_m, \omega)
# \end{align*}
# is the machine learning model with parameters $\omega$ to be determined by minimizing $R_M$. 
# The quantity $x =  m_{hh}, c_{tt}, c_{ggh}, c_{gghh}$ are the inputs to the model and the target $t$ is the predicted cross section (per 15 GeV in $m_{hh}$).
# 
# (Aside: The empirical risk $R_M$ approximates the **risk functional**
# \begin{align}
# R[f] & = \int \cdots \int \, p(t, \boldsymbol{x}) \, L(t, f(\boldsymbol{x}, \omega)) \, dt \, d\boldsymbol{x} ,
# \end{align}
# where the quantity $p(t, \boldsymbol{x}) \, dt\, d\boldsymbol{x}$ is the probability distribution of the training data from which the sample $\{ (t_m, \boldsymbol{x}_m), m = 1,\cdots, M \}$ is presumed to have been drawn.) 
# 
# We shall fit the model $f$ for the cross section by minimizing the **weighted quadratic loss**
# 
# \begin{align}
#     L(t, f) &= w (t - f)^2 ,
# \end{align}
# where $f$ is a deep neural network and the weight is set to $w = 1 / t$. Weighting each loss term so that each term contributes roughly the same amount in the empirical risk.  This was inspired by Nicola's brilliant suggestion to use the errors in each histogram bin!
# 
# If 1) the function $f$ has sufficient capacity (i.e., there exists a choice of parameters that yield an approximation arbitrarily close to the exact function $\sigma(m_{hh}, \theta)$), and 2) we have enough training data, and 3) the minimizer can find a good approximation to the minimum of the risk functional, then the calculus of variations shows that the minimum of the quadratic loss leads occurs when
# \begin{align}
#     f(\boldsymbol{x}, \omega^*) & = \int t \, p(t | \boldsymbol{x}) \, dt ,
# \end{align}
# where $\omega^*$ denotes the best-fit value of $\omega$, $p(t | \boldsymbol{x}) = p(t,  \boldsymbol{x}) / p(\boldsymbol{x})$, and $p(t, \boldsymbol{x})$ is the (typically, *unknown*) probability distribution of the training data.

# ### Define model for cross section
# `heftnet_direct` models the mapping $f : m_{hh}, c_{tt}, c_{ggh}, c_{gghh} \rightarrow \sigma$.

# In[41]:

open('heftnet_direct.py', 'w').writelines(["import torch\nimport torch.nn as nn\nimport numpy as np\n\nname     = 'heftnet_direct'\nfeatures = ['mhh', 'CTT', 'CGGH', 'CGGHH']\ntarget   = 'target'\nnodes    = 15\nnhidden  = 12\nnoutputs =  1\n\nclass Sin(nn.Module):\n\n    def __init__(self):\n        # initial base class (nn.Module)\n        super().__init__()\n\n    def forward(self, x):\n        return torch.sin(x)\n\nclass ResNet(nn.Module):\n\n    def __init__(self):\n        # initial base class (nn.Module)\n        super().__init__()\n        self.NN = nn.Sequential(nn.Linear(nodes, nodes), nn.ReLU(),\n                                nn.Linear(nodes, nodes), nn.ReLU())    \n    def forward(self, x):\n        return self.NN(x) + x\n\nclass HEFTNet(nn.Module):\n\n    def __init__(self):\n\n        # initial base class (nn.Module)\n        super().__init__()\n\n        cmd = 'self.xsec = nn.Sequential(nn.Linear(len(features), nodes), nn.SiLU(),'\n        \n        for _ in range(nhidden):\n            cmd += 'nn.Linear(nodes, nodes), nn.SiLU(),'\n            \n        cmd += 'nn.Linear(nodes, noutputs))'\n        \n        exec(cmd)\n        \n    # required method: this function computes the sqrt(cross section)\n    def forward(self, x):\n        # x.shape: (N, 4), where N is the batch size\n        return self.xsec(x)\n"])

# In[42]:


import heftnet_direct as NN
importlib.reload(NN)

name     = NN.name
model    = NN.HEFTNet()
features = NN.features
target   = NN.target

modelfile  = '%s.dict' % NN.name
print(name)
print(model)
print('number of parameters: %d\n' % dn.number_of_parameters(model))

# check model
X = torch.Tensor(test_data[['mhh','CTT', 'CGGH', 'CGGHH']].to_numpy())
print('input.size:  ', X.size())

Y = model(X)
print('output.size: ', Y.size())


# ### Weighted quadratic loss

# In[43]:


def average_quadratic_loss_weighted(f, t, x=None):
    # f and t must be of the same shape
    w = torch.where(t != 0, 1/torch.abs(t), 1)
    return  torch.mean(w * (f - t)**2)


# ### Train!

# In[44]:


traces = ([], [], [])


# In[46]:


import dnnutil as dn
importlib.reload(dn)

traces_step   = 1000
n_batch       = 150
n_iterations  = 2000000
early_stopping= 200000
learning_rate = 2.e-4

print('='*60)
print(f'trace step:           {traces_step:10d}')
print(f'batch size:           {n_batch:10d}')
print(f'number of iterations: {n_iterations:10d}')
print(f'early stopping count: {early_stopping:10d}')
print(f'learning rate:        {learning_rate:10.2e}')
print('='*60)

# In[47]:


av_loss = average_quadratic_loss_weighted

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

traces = dn.train(model, optimizer, 
                  modelfile, early_stopping,
                  av_loss,
                  dn.get_batch, 
                  train_data, valid_data,
                  features, target,
                  n_batch, 
                  n_iterations,
                  traces,
                  step=traces_step)

print('\nDone with training!\n')
logdf = pd.DataFrame(np.array(traces).T, columns=['iteration', 'trainloss', 'validloss'])
logdf.to_csv(f'{name}_loss.csv')
