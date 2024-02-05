import networkx as nx
import pandas as pd
from pdb import set_trace

def evaluate_welfare(gd, T):
    val = 0
    for (b,bid,s,sid) in T:
        val += gd.f_b(b,bid) - gd.f_s(s, sid)

    return val

def agent_tot_value(gd, i):
    val = 0
    if gd.is_seller(i):
        fun = gd.f_s
    else:
        fun = gd.f_b
    for j in range(1,gd.k+1):
        val += fun(i, j)
    return val

def evaluate_normalized_total_value(gd, T):
    welfare = evaluate_welfare(gd, T)

    sigma_0 = 0
    sigma_seller = 0
    for i in gd.compatibility_graph.nodes:
        agent_val = agent_tot_value(gd, i)
        sigma_0 += agent_val
        if gd.is_seller(i):
            sigma_seller += agent_val

    val1 = (sigma_seller + welfare)/sigma_0
    val2 = welfare/sigma_0
    ## if val > 1:
    ##     set_trace()
    return val1, val2

def resources_needs_graph(gd):
    cgraph = gd.get_compatibility_graph()
    buyers = [node for node, attributes in cgraph.nodes(data=True) if attributes.get('bipartite') == 0]
    sellers = [node for node, attributes in cgraph.nodes(data=True) if attributes.get('bipartite') == 1]

    g = nx.Graph()

    for buyer in buyers:
        requirement = gd.get_rho(buyer)
        for item in range(1,requirement+1):
            g.add_node("{}_{}".format(buyer, item), bipartite=0, buyer=buyer, bid=item)

    for seller in sellers:
        availability = gd.get_rho(seller)
        for item in range(1,availability+1):
            g.add_node("{}_{}".format(seller, item), bipartite=1, seller=seller, sid=item)

    bunits = [node for node, attributes in g.nodes(data=True) if attributes.get('bipartite') == 0]
    sunits = [node for node, attributes in g.nodes(data=True) if attributes.get('bipartite') == 1]

    for bu in bunits:
        for su in sunits:
            if g.nodes[bu]['buyer'] in cgraph.neighbors(g.nodes[su]['seller']):
                weight = gd.f_b(g.nodes[bu]['buyer'], g.nodes[bu]['bid']) - gd.f_s(g.nodes[su]['seller'],g.nodes[su]['sid'])
                if weight > 0:
                    g.add_edge(bu, su, weight=weight)

    records = []
    for edge in g.edges():
        u,uid = [int(ele) for ele in edge[0].split('_')]
        v,vid = [int(ele) for ele in edge[1].split('_')]
        if gd.is_seller(u):
            s,sid = (u,uid)
            b,bid = (v,vid)
        else:
            s,sid = (v,vid)
            b,bid = (u,uid)

        records.append([s,sid,b,bid,g.edges[edge]['weight']])

    return pd.DataFrame.from_records(records, columns=[
        'seller','seller_unit','buyer','buyer_unit','weight'])
