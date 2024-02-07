#!/usr/bin/env python
DESC = '''
Main file to run real networks.
'''

import argparse
import csv
import glob
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
from re import sub
import time
from pdb import set_trace

from max_welfare_fairness import optimize_welfare

directory_path = ''
#dataset = 'touchet'
#delta = 10
#acre_feet = 20

def parser():
    # parser
    parser=argparse.ArgumentParser(description=DESC,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--network', required=True, help='Network')
    parser.add_argument('-d', '--delta', required=True, type=int,
            help='Water availability')
    parser.add_argument('-a', '--acre_feet', required=True, type=int,
            help='Acre feet')
    parser.add_argument('-r', '--req', required=True, type=float,
            help='Min. requirement in percentage')
    return parser.parse_args()

def generate_graph(dataset, drought, acre_feet):
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

    if len(csv_files) > 1:
        raise('More than one file selected')
    data = pd.read_csv(csv_files[0])
    data = data.rename(columns={
        'seller_i': 'seller_unit',
        'buyer_i': 'buyer_unit'})
    data['weight'] = data.buyer_budget_per_unit - data.seller_value_per_unit
    data = data[data.weight>0]
    return data

if __name__ == '__main__':
    args = parser()
    net = generate_graph(args.network, args.delta, args.acre_feet)
    req = net[['buyer', 'buyer_unit']].groupby('buyer').count().reset_index()
    req['requirement'] = np.ceil(req.buyer_unit * args.req / 100)
    req.requirement = req.requirement.astype('int')

    obj = optimize_welfare(net, req)
    req['requirement'] = 0
    obj0 = optimize_welfare(net, req)
    try:
        mw_fairness = obj.ObjVal
    except:
        mw_fairness = 0
    print( f"{args.network.capitalize()}{args.acre_feet}", args.network, args.delta, args.acre_feet, args.req, mw_fairness, obj0.ObjVal, sep=',')
