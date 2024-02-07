DESC = '''
Main file to run real networks.
'''

from re import sub
import pandas as pd
import glob
import csv
import os
import time
import networkx as nx
import numpy as np

directory_path = ''

def generate_graph(arg):
    dataset = arg[0]
    drought = arg[1]
    acre_feet = arg[2]

    g = nx.Graph()
    # Use glob to get a list of all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, dataset, str(drought), str(acre_feet), '*.csv'))

    # Initialize an empty dictionary
    drought_additive = {}

    stats = os.path.join(directory_path, dataset, "{}_statistics_by_water_capacity.csv".format(dataset.capitalize()))


    # Open the CSV file and read its contents
    with open(stats, mode='r') as file:
        first = True
        for line in file:
            if first:
                first = False
                continue

            sp = line.split(",")
            if int(sp[1]) == acre_feet:
                drought_additive[float(sp[0])] = float(sp[2])
            if int(sp[1]) == acre_feet and int(sp[0])==drought:
                buyer_count = int(sp[4][:-1])
    print(drought_additive)
    denominator = drought_additive[100.0]

    # Iterate over the list of CSV files
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        # Create the first table: col2 and the greatest value of col1 for each unique value in col2
        table1 = data.groupby('seller')['seller_i'].max().reset_index()
        table1.columns = ['seller', 'max_id']

        table2 = data.groupby('buyer')['buyer_i'].max().reset_index()
        table2.columns = ['buyer', 'max_id']

        srhos = table1.set_index('seller')['max_id'].add(1).to_dict()
        brhos = table2.set_index('buyer')['max_id'].add(1).to_dict()

        buyer_units = {}
        with open(csv_file, 'r') as file:
            # Use the csv.reader to parse the CSV file
            csv_reader = csv.reader(file)
            next(csv_reader)
            sellers_set = set()
            buyers_set = set()
            # You can now iterate over rows in the CSV file
            for rind, row in enumerate(csv_reader):
                # Do something with each row, for example, print it
                node_1 = row[4] + "-" + row[3]
                node_2 = row[1] + "-" + row[0]

                ## sellers_set.add(row[1])
                ## buyers_set.add(row[4])

                value_seller = row[2]
                value_buyer = row[5]

                g.add_node(node_1, bipartite=0, rho=brhos[int(row[4])], weight=float(value_buyer))
                g.add_node(node_2, bipartite=1, rho=srhos[int(row[1])], weight=float(value_seller))
                try:
                    buyer_units[int(row[4])].add(int(row[3]))
                except:
                    buyer_units[int(row[4])] = set([int(row[3])])

                if float(value_buyer) - float(value_seller) < 0:
                    raise Exception("PROBLEMATIC EDGE")

                g.add_edge(node_1, node_2, weight=float(value_buyer) - float(
                    value_seller))  # buyer, seller, value for buyer , value for seller

                if rind % 100000 == 0:
                    print("done ", rind)

    buyer_deg = {x: len(y)  for x,y in buyer_units.items()}
    print("done creating G")

    # Get the nodes from the left and right partitions
    left_nodes = [node for node, data in g.nodes(data=True) if data["bipartite"] == 0]
    right_nodes = [node for node, data in g.nodes(data=True) if data["bipartite"] == 1]

    # Create an empty adjacency matrix
    adjacency_matrix = np.zeros((len(left_nodes), len(right_nodes)))

    # Fill the adjacency matrix with edge weights
    for i, left_node in enumerate(left_nodes):
        for j, right_node in enumerate(right_nodes):
            if g.has_edge(left_node, right_node):
                adjacency_matrix[i, j] = g[left_node][right_node]["weight"]
    print("have a graph ")

    t = time.time()
    # Solve the linear sum assignment problem
    mwm = nx.max_weight_matching(g)
    buyers = [node for node, data in g.nodes(data=True) if data["bipartite"] == 0]
    mbuyers = {}
    msellers = {}
    for edge in mwm:
        if edge[1] in buyers:
            mbuyers[edge[1]] = edge[0]
            msellers[edge[0]] = edge[1]
        else:
            mbuyers[edge[0]] = edge[1]
            msellers[edge[1]] = edge[0]

    matched_buyers = list(mbuyers.keys())
    matched_sellers = list(msellers.keys())

    print("before existence loop")
    exists = True
    while exists:
        exists = False
        for b in matched_buyers:
            sp = b.split("-")
            b1 = sp[0]
            b2 = sp[1]
            if int(b2) > 0:
                earlier = "-".join([b1, str(int(b2) - 1)])
                if earlier not in mbuyers.keys():
                    s = mbuyers[b]
                    mbuyers[earlier]= s
                    msellers[s] = earlier
                    del mbuyers[b]
                    exists = True

        if exists:
            continue
        for s in matched_sellers:
            sp = s.split("-")
            s1 = sp[0]
            s2 = sp[1]
            if int(s2) > 0:
                earlier = "-".join([s1, str(int(s2) - 1)])
                if earlier not in msellers.keys():
                    b = msellers[s]
                    mbuyers[b] = earlier
                    msellers[earlier] = b
                    del msellers[s]
                    exists = True
        if exists:
            matched_buyers = list(mbuyers.keys())
            matched_sellers = list(msellers.keys())
    print("after existence loop")

    weight= 0
    buyer_sat = {}
    for buyer, seller in mbuyers.items():
        weight += g[buyer][seller]['weight']
        b = int(sub('-.*', '', buyer))
        try:
            buyer_sat[b] += 1
        except:
            buyer_sat[b] = 1

    weight2 = drought_additive[drought] / denominator
    norm_weight = (weight + drought_additive[drought]) / denominator

    df = pd.DataFrame.from_dict(buyer_deg, orient='index')
    df = df.rename(columns={0: 'tot'})
    df1 = pd.DataFrame.from_dict(buyer_sat, orient='index')
    df1 = df1.rename(columns={0: 'sat'})

    df = df.join(df1).fillna(0)
    df['satisfaction'] = df.sat / df.tot

    df['delta'] = drought
    df['acre_feet'] = acre_feet
    df['data'] = dataset
    df['network'] = df.data.str.capitalize() + df.acre_feet.astype('str')
    df = df.reset_index().rename(columns={'index': 'buyer'})
    st = ",".join(map(str, [f"{dataset.capitalize()}{acre_feet}", dataset, drought, acre_feet, len(g.edges()), len(g.nodes()), len(mbuyers.keys()), len(msellers.keys()), weight, weight2,  norm_weight, denominator]))
    return st,df

if __name__ == '__main__':
    args = []
    for dataset in {"yakima", "touchet" }:
        for acre in {5,10,20}:
            for delta in range(10,100,10):
                args.append((dataset,delta,acre))

    ret = map(generate_graph, args)
    with open(os.path.join("results","results_realdata.csv"),"w") as f:
        print("network,data,delta,acre,num_edges,num_nodes,num_buyers,num_sellers,welfare,sellers_value_by_tot_value,sigma_T_by_sigma_0,sigma_0",file=f,flush=True)
        dfs = []
        for item,df in ret:
            print(item, file=f, flush=True)
            dfs.append(df)

        concatenated_df = pd.concat(list(dfs), ignore_index=True)
        concatenated_df.to_csv(os.path.join('results','sat.res.csv'), index=False)
