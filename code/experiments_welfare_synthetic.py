#!/usr/bin/env python
import numpy as np
import pandas as pd
from pdb import set_trace
import synthetic
import util
from algorithm1 import algorithm1

N = 10

zp = []
for i in range(100):
    for delta in [0.3,0.5,0.7,0.9]:
        for gamma in [0, 0.5, 1]:
            for beta_high in [0.7]:
                G = synthetic.TradingSyntheticGraph(gamma=gamma, beta_low=1-beta_high, beta_high=beta_high, delta=delta)
                num_sellers = 0
                for node in G.compatibility_graph.nodes:
                    num_sellers += G.is_seller(node)

                T, num_edges = algorithm1(G)

                x,y = util.evaluate_normalized_total_value(G, T) 

                zp.append((i, beta_high, delta, num_sellers, x, y, util.evaluate_welfare(G,T), gamma,num_edges))


df = pd.DataFrame(zp, columns= ['replicate', 'beta_high', 'delta', 'num_sellers', 'total_value', 'normalized_welfare', 'welfare', 'lambda', 'num_edges'])

df.to_csv('results_synthetic.csv', index=False)

