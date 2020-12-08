---
layout: default
title: FAQ and Vocabulary Guide
nav_order: 3
permalink: /FAQ
---


# FAQ and Vocabulary Guide
{: .no_toc }

---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Vocabulary Guide

We briefly define some of the terms we use interchangeably throughout this User Guide and in this documentation below.

| Unit, Observation, Individual | A participant in the research trial, in either the control group or treatment group, for whom we have an observed outcome                     |
| Covariate, Observed data, X's, Independent variables  | The data we observe which is not the treatment group or the outcome      |
|  Outcome, Y, Dependent variables               | The outcome variable of the research |
| Treated Unit | A unit which is in the treatment group |
| Treatment Effects | The effect on the outcome by the treatment. |
| Matched group, matches | The list of units a particular covariate is matched with, including their weights|



## FAQ

### What kind of covariates can the dataset have?
MALTS works on continuous, categorical or mixed covariates.

### Why doesn't the package have any built-in visualization methods?
Visualizing data is a valuable tool to understanding it before and after performing any analysis like matching. While the `PYMALTS2` package doesn't have built in functions to visualize the data, we provide several examples of ways that researchers could visualize any dataset.

### Why should I use this instead of another package? Other ones seem more common!

The matches produced by the `Pymalts2` package are higher quality. `Pymalts2` doesn't use uninterpretable propensity scores, it matches on actual covariates. It doesn't require the user to specify the metric like CEM, since it uses machine learning to learn the metric adaptively. It is not based on a black box machine learning method like Causal Forest or BART, but it can often be just as accurate, and it’s much easier to troubleshoot! <a href="#references">[1]</a>. 

### I have a question not covered here

Please reach out to let our team know if you’re using this, or if you have any questions. Contact Harsh Parikh at harsh.parikh@duke.edu

<div id="references" class="language-markdown highlighter-rouge">
  <h4>References</h4>
  <a class="number" href="#flame">[1]</a>
  <a href="https://arxiv.org/abs/1811.07415">
    Parikh, Rudin, et al. <i>MALTS: Matching After Learning to Stretch</i>.
  </a> 
  <br/>
</div>