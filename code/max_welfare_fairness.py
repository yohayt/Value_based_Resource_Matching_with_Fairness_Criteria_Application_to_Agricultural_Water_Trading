import gurobipy as gp
from gurobipy import GRB, quicksum
import networkx as nx
import numpy as np
import pandas as pd

## with gp.Env(empty=True) as env:
##     env.setParam("WLSAccessID", str)
##     env.setParam("WLSSECRET", str)
##     env.setParam("LICENSEID", int)
##     env.setParam("OutputFlag", 0)
##     env.start()

# returns a Gurobi model object
def optimize_welfare(comp_graph_df, buyer_req_df):
    buyer_req_df = buyer_req_df.set_index('buyer')
    
    # Read into a networkX graph
    G = nx.Graph()

    Vs = []
    Vb = []
    for idx, row in comp_graph_df.iterrows():
        sell_node = str(int(row['seller']))+'_'+str(int(row['seller_unit']))
        buy_node = str(int(row['buyer']))+'_'+str(int(row['buyer_unit']))

        if sell_node not in Vs:
            Vs.append(sell_node)

        if buy_node not in Vb:
            Vb.append(buy_node)

        # Don't add edges with 0 or negative weights
        if row['weight'] <= 0:
            continue

        G.add_edge(sell_node, buy_node, weight=row['weight'])
    
    # Extract buyer id's
    B = []
    for node in Vb:
        buyer = int(node.split('_')[0])
        if buyer not in B:
            B.append(buyer)
    
    
    welfare_model = gp.Model("lp_welfare")

    Z = welfare_model.addVars(G.edges(), vtype=GRB.CONTINUOUS)

    welfare_model.update()

    for edge,z_var in Z.items():
        welfare_model.addConstr(z_var>=0)

    for node in Vs:
        welfare_model.addConstr(quicksum([z_var for edge, z_var in Z.items() if node in edge]) <= 1)

    for node in Vb:
        welfare_model.addConstr(quicksum([z_var for edge, z_var in Z.items() if node in edge]) <= 1)

    for buyer in B:
        req = buyer_req_df.loc[buyer].requirement
        welfare_model.addConstr(quicksum([z_var for edge, z_var in Z.items() if buyer == int(edge[1].split('_')[0]) or buyer == int(edge[0].split('_')[0])]) >= req)

    welfare_model.setObjective(quicksum(z_var*G.get_edge_data(edge[0],edge[1])['weight'] for edge, z_var in Z.items()), GRB.MAXIMIZE)
    
    welfare_model.optimize()
    
    return welfare_model
    
    
    
