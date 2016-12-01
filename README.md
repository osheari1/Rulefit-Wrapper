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
