---
layout: default
title: Installation and Quickstart Example
nav_order: 1
permalink: /getting-started/installation_example/
parent: Getting Started
---
# Installation and Quickstart Example

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Dependencies
This package requires prior installation of
- Python (>= 3.0)
- NumPy (>= 1.17.5)
- Scikit-Learn (>= 0.22.1))
- Pandas (todo: check)
- Matplotlib
- Seaborn

If your computer system does not have python 3.*, install from [here](https://www.python.org/downloads/).

If your python version does not have the Pandas, Scikit learn, or Numpy packages, install from [here](https://packaging.python.org/tutorials/installing-packages/)

## Installation
The MALTS Python Package is available for download on the [almost-matching-exactly Github](https://github.com/almost-matching-exactly/MALTS) 
or via PyPi (recommended):

{% highlight markdown %}
pip install pymalts2
{% endhighlight %}

## Quickstart Example

We show the working of the package. In this example, we provide only the basic inputs: (1) input data as a dataframe or file, (2) the name of the outcome column, and (3) the name of the treatment column.
In order to set up the model for learning the distance metric, we consider:

1. Variable name for the outcome variable: 'outcome'.
2. Variable name for the treatment variable: 'treated'
3. Data is assigned to python variable df

<div class="code-example" markdown="1">
```python
import pandas as pd
import pymalts2 as pymalts
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('example/example_data.csv',index_col=0)
print(df.shape)
df.head()

#> (2500, 20)
#>	    X1	        X2	        X3 	        X4	        X5	        X6	        X7	        X8	        X9	        X10  	           X11	            X12 	      X13	      X14	      X15	       X16	      X17	     X18	   outcome   	  treated
#>1355	1.881335	1.684164	0.532332	2.002254	1.435032	1.450196	1.974763	1.321659	0.709443	-1.141244	0.883130	0.956721	2.498229	2.251677	0.375271	-0.545129	3.334220	0.081259	-15.679894	0
#>1320	0.666476	1.263065	0.657558	0.498780	1.096135	1.002569	0.881916	0.740392	2.780857	-0.765889	1.230980	-1.214324	-0.040029	1.554477	4.235513	3.596213	0.959022	0.513409	-7.068587	0
#>1233	-0.193200	0.961823	1.652723	1.117316	0.590318	0.566765	0.775715	0.938379	-2.055124	1.942873	-0.606074	3.329552	-1.822938	3.240945	2.106121	0.857190	0.577264	-2.370578	-5.133200	0
#>706	1.378660	1.794625	0.701158	1.815518	1.129920	1.188477	0.845063	1.217270	5.847379	0.566517	-0.045607	0.736230	0.941677	0.835420	-0.560388	0.427255	2.239003	-0.632832	39.684984	1
#>438	0.434297	0.296656	0.545785	0.110366	0.151758	-0.257326	0.601965	0.499884	-0.973684	-0.552586	-0.778477	0.936956	0.831105	2.060040	3.153799	0.027665	0.376857	-1.221457	-2.954324	0

m = pymalts.malts_mf( outcome='outcome', treatment='treated', data=df) # running MALTS with default setting
```
</div>

## Matched Groups

Matched Group matrix (MG_matrix) is NxN matrix with each row corresponding to each query unit and each column corresponds to matched units. Cell (i,j) in the matrix corresponds to the weight of unit j in the matched group of unit i. The weight corresponds to the numbers of times a unit is included in a matched group across M-folds.

The `CATE_df` dataframe in the model `m` gives us the CATE estimate for a corresponding unit in each row.
<div class="code-example" markdown="1">
```python
print (m.CATE_df)
#>	avg.CATE	std.CATE	outcome	   treated
#>0	47.232061	21.808950	-15.313091	0.0
#>1	40.600643	21.958906	-16.963202	0.0
#>2	40.877320	22.204570	9.527929	1.0
#>3	37.768578	19.740320	-3.940218	0.0
#>4	39.920257	21.744433	-8.011915	0.0
#>...   	...    	...	    ...    	...
#>2495	49.227788	21.581176	-14.529871	0.0
#>2496	42.352355	21.385861	19.570055	1.0
#>2497	43.737763	19.859275	-16.342666	0.0
#>2498	41.189297	20.346711	-9.165242	0.0
#>2499	45.427037	23.762884	-17.604829	0.0

ATE = m.CATE_df['avg.CATE'].mean()
print (ATE)
#>42.29673993471417
```
</div>