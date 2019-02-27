#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:31:59 2018

@author: harshparikh
"""

def dnaComplement(s):
    s1 = ''
    for i in s:
        if i == 'A':
            s1 = s1+'T'
        elif i == 'T':
            s1 = s1+'A'
        elif i == 'C':
            s1 = s1+'G'
        elif i == 'G':
            s1 = s1+'C'
    s2 = ''
    for i in s1:
        s2 = i + s2
    return s2   

def budgetShopping(n, bundleQuantities, bundleCosts):
    m = min(bundleCosts)
    if m>n:
        return 0
    a = 0
    for i in range(0,len(bundleCosts)):
        if bundleCosts[i]<n:
            b = bundleQuantities[i] + budgetShopping(n-bundleCosts[i],bundleQuantities,bundleCosts)
            if b>=a:
                a = b
    return a

def deleteProducts(ids,m):
    def setup(ids):
        d = {}
        for i in ids:
            if i not in d:
                d[i] = 0
            d[i] = d[i] + 1
        d_sort_val = sorted(d.items(), key=lambda kv: kv[1])
        return d_sort_val
    dsv = setup(ids)
    def dp(dsv,m):
        if m <= 0:
            return dsv
        dn = min(m,dsv[0][1])
        dsv[0] = (dsv[0][0],dsv[0][1] - dn)
        if dsv[0][1]<=0:
            del dsv[0]
        return dp(dsv,m-dn)
    return len(dp(dsv,m))
