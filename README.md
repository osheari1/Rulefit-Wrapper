# Rulefit
Python wrapper for R rulefit package

## Table of Contents
1. Installation Instructions[1]
2. [2]
3. [3]
4. Examples[4]
### 1. Installation Instructions
### 2.
### 3.
### 4. Examples
All the Rulefit operations are controled through the Rulefit object.
```python
from rulefit import Rulefit

rfhome = '/path/to/R/rulefit/dir/'
platform = 'linux'

model = Rulefit(platform, rfhome)
```
Fit a Rulefit model by calling the fit function. 
```python
X = pd.read_csv('path/to/X')
y = pd.read_csv('path/to/y')
model.fit(x=X, y=y, rfmode='class', tree_size=5, mod_sel=3)
print(model.rules)
```


```python
import numpy as np
import pandas as pd
boston = pd.read_csv('./datasets/boston.csv', index_col=False)
boston['target'] = np.select([boston.medv > boston.medv.quantile(0.5)],
                             [1], [-1])
boston['lstat_cat'] = pd.cut(boston.lstat, 10,
                             labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                                     'i', 'j'])
boston.drop('lstat', axis=1, inplace=True)

rf_path_w = '/home/riley/R/x86_64-pc-linux-gnu-library/3.3/Rulefit'
rf_path_h = '/home/riley/R/x86_64-pc-linux-gnu-library/3.3/rulefit'

model = RuleFit('linux', rf_path_h, './logs/rulefit.log')
model.fit(x=boston.drop(['medv', 'target'], axis=1),
          y=boston['target'],
          rfmode='class', tree_size=5, mod_sel=3,
          max_rules=500)

model._generate_rules()
# pprint(model._rules)
# model.single_partial_dependency(list(range(model.data['x'].shape[1])),
# nav=1000)
model.double_partial_dependencies(1, 3, plot_type='contour')

# model.generate_intr_effects(nval=100, n=10, quiet=False, plot=True)

# two_var_int = model.two_var_intr_effects(
# target='rm',
# vars=list(set(model.data['x'].columns.values).difference(['rm'])))

# thr_var_int = model.three_var_intr_effects(
# tvar1='rm',
# tvar2='dis',
# vars=list(set(model.data['x'].columns.values).difference(['rm', 'dis'])))

```
