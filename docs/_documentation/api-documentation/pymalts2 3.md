---
layout: default
title: MALTS
nav_order: 1
permalink: /api-documentation/pymalts
parent: API Documentation
---

# Malts_MF Class

<div class="code-example" markdown="1">
```python
class pymalts.malts_mf(outcome,treatment,data,discrete=[],C=1,k_tr=15,k_est=50,
	 estimator="linear",smooth_cate=True,reweight=False,n_splits=5,
	 n_repeats=1,output_format="brief")  
```
</div>

<div id="source" class="language-markdown highlighter-rouge">
  <a class="number" href="#SourceCode"></a> 
  <a href="https://github.com/almost-matching-exactly/MALTS/blob/master/pymalts.py">
    <h6><u>Source Code</u></h6>
  </a>
</div>
This class creates the matches based on the MALTS: "Matching After Learning to Stretch" algorithm. It has built in support for stopping criteria and missing data handling. 

## Parameters

### Required Parameters

| Parameter Name   | Type                                        | Default | Description                                                         |
|------------------|---------------------------------------------|---------|---------------------------------------------------------------------|
|data|file, Dataframe|required	|The data to be matched. Preferably, the data should be in the form of a Python Pandas DataFrame. |
|outcome|string|required |The column name containing the name of the outcome variable, which itself is numeric.|
|treated|string|required|The column name denoting whether the unit is treated or control for the matching procedure. |

### Optional Parameters

|discrete|list| [ ]|The list of columns that have been dummified (discrete).|
|C|integer| 1|The regularization constant used in the objective method with the matrix.|
|k_tr|integer|15|The size of the matched group in the training step.|
|k_est|integer|50|The size of the matched group in the estimation step.|
|estimator|string| 'linear'|The method used to estimate the CATE value inside a matched group. The possible options are 'linear', 'mean' or 'RF', which use ridge regression, mean regression, and Random Forest regression, respectively.|
|smooth_cate|boolean|True|Boolean to specify whether the CATE estimates should be smoothened by using a regression model to obtain a fit.|
|reweight|boolean|False	|Boolean to specify if treatment and control groups should be reweighted as per their sample sizes in the training step.|
|n_splits|integer|  5|	The number of splits of the data when n_split-fold procedure is used.|
|n_repeats|integer|  1	|The number of times the whole procedure is repeated.|
|output_format|string|'brief'	|The style in which the output CATE dataframe is to be displayed. Possible options are 'brief' and 'full'. If 'full' is chosen, the entire dataframe is displayed, if 'brief' is chosen, only the columns 'avg_CATE', 'std_CATE', 'outcome', and 'treatment' are displayed.|


