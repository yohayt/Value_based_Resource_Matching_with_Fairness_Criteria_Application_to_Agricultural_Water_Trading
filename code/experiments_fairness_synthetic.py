#!/usr/bin/env python
import numpy as np
import pandas as pd
from pdb import set_trace
import synthetic
import util
from algorithm1 import algorithm1

from max_welfare_fairness import optimize_welfare

N = 10

zp = []
for i in range(100): #range(100):
    for delta in np.arange(0.1,1,0.1):
        for gamma in [0, 0.5, 1]:
            for beta_high in [0.7, 0.9]:
                G = synthetic.TradingSyntheticGraph(gamma=gamma, 
                        beta_low=1-beta_high, beta_high=beta_high, 
                        delta=delta)
                rng = util.resources_needs_graph(G)
                netname = f'rng_rep{i}_d{delta}_g{gamma}_b{beta_high}'
                #rng.to_csv(f'{netname}.csv', index=False)
                for rb in [1, 2, 3, 4, 5]:
                    req = rng.buyer.drop_duplicates().to_frame()
                    req['requirement'] = rb
                    obj = optimize_welfare(rng, req)
                    req['requirement'] = 0
                    obj0 = optimize_welfare(rng, req)
                    try:
                        mw_fairness = obj.ObjVal
                    except:
                        mw_fairness = 0
                    #T, __ = algorithm1(G)
                    #mw_alg1 = util.evaluate_welfare(G,T)
                    
                    ## req.to_csv(f'req_{rb}_{netname}.csv', index=False)
                    zp.append([i, delta, gamma, beta_high, rb, mw_fairness, obj0.ObjVal])
                
df = pd.DataFrame(zp, columns= ['replicate', 'delta', 'lambda', 'beta_high', 'rb', 'welfare', 'welfare0'])

df.to_csv('results_fairness.csv', index=False)

