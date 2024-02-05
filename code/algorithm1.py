import networkx as nx
import pandas as pd
from pdb import set_trace

from synthetic import TradingSyntheticGraph
import util

def algorithm1(gd):
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
                g.add_edge(bu, su, weight=gd.f_b(g.nodes[bu]['buyer'], g.nodes[bu]['bid']) - gd.f_s(g.nodes[su]['seller'],g.nodes[su]['sid']))

    num_edges = 0
    for edge in g.edges:
        if g.edges[edge]['weight'] >= 0:
            num_edges += 1

    M = nx.max_weight_matching(g, weight='weight')
    #welf=sum([g.edges[e]['weight'] for e in M])

    fbunits = {}
    for unit in bunits:
        buyer = g.nodes[unit]['buyer']
        bid = g.nodes[unit]['bid']
        if buyer not in fbunits:
            fbunits[buyer] = [bid]
        else:
            fbunits[buyer].append(bid)
    for buyer in fbunits:
        fbunits[buyer].sort()

    fsunits = {}
    for unit in sunits:
        seller = g.nodes[unit]['seller']
        sid = g.nodes[unit]['sid']
        if seller not in fsunits:
            fsunits[seller] = [sid]
        else:
            fsunits[seller].append(sid)
    for seller in fsunits:
        fsunits[seller].sort()

    T = set()
    for (x, y) in M:
        if 'buyer' in g.nodes[x]:
            T.add((g.nodes[x]['buyer'], g.nodes[x]['bid'], g.nodes[y]['seller'], g.nodes[y]['sid']))
        else:
            T.add((g.nodes[y]['buyer'], g.nodes[y]['bid'], g.nodes[x]['seller'], g.nodes[x]['sid']))

    for (b, bid, s, sid) in T:
        fbunits[b].remove(bid)
        fsunits[s].remove(sid)

    r1 = -1
    while r1 is not None:
        r1 = None
        r2 = None
        for (b, bid, s, sid) in T:
            if bid - 1 in fbunits[b]:
                r1 = (b, bid, s, sid)
                r2 = (b, bid - 1, s, sid)
                break
            if sid - 1 in fsunits[s]:
                r1 = (b, bid, s, sid)
                r2 = (b, bid, s, sid - 1)
                break
        if r1 is not None:
            T.remove(r1)
            T.add(r2)
    return T, num_edges

## delta=  False
## beta = False
## N= False
## gamma = False
## betadiff = False
## heatmap = True
## 
## if delta:
##     zp = []
##     for i in range(100):
##         for gamma in {0.1,0.5,0.9}:
##             for delta in range(0,11):
##                 gd = TradingSyntheticGraph(delta=delta/10, gamma=gamma)
##                 zp.append((delta/10, util.evaluate_welfare(gd, algorithm1(gd)),gamma))
## 
##     df = pd.DataFrame(zp, columns= ['δ','Social Welfare','γ'])
## 
##     sns.lineplot(data=df[df['γ'] == 0.1], x='δ', y='Social Welfare', label='γ=0.1')
## 
##     # Create a line plot for 'B'
##     sns.lineplot(data=df[df['γ'] == 0.5], x='δ', y='Social Welfare', label='γ=0.5')
## 
##     # Create a line plot for 'C'
##     sns.lineplot(data=df[df['γ'] == 0.9], x='δ', y='Social Welfare', label='γ=0.9')
## 
##     plt.xlabel('δ', fontsize=15)
##     plt.ylabel('Social Welfare', fontsize=15)
##     plt.title('', fontsize=15)
## 
##     # Add grid lines
##     plt.grid(True, linestyle='--', alpha=0.5)
## 
##     # Add a legend with increased font size
##     plt.legend(fontsize=15)
## 
##     # Save the plot as a PDF file
##     plt.savefig('delta.pdf', format='pdf')
## 
## 
## ###################################################
## 
## if gamma:
##     plt.clf()
## 
##     zp = []
##     for i in range(100):
##         for gamma in range(0,11):
##             gd = TradingSyntheticGraph(gamma=gamma/10)
##             zp.append((gamma/10, util.evaluate_welfare(gd, algorithm1(gd))))
## 
## 
##     # Extract x and y coordinates from the pairs
##     x_values, y_values = zip(*zp)
##     df = pd.DataFrame({'gamma': x_values, 'SW': y_values})
##     sns.lineplot(x='gamma', y='SW', data=df, marker='o', ci='sd', err_style='bars', color='blue', label='Algorithm 1')
## 
##     plt.xlabel('Gamma', fontsize=15)
##     plt.ylabel('Social Welfare', fontsize=15)
##     plt.title('', fontsize=15)
## 
##     # Add grid lines
##     plt.grid(True, linestyle='--', alpha=0.5)
## 
##     # Add a legend with increased font size
##     plt.legend(fontsize=15)
## 
##     # Save the plot as a PDF file
##     plt.savefig('gamma.pdf', format='pdf')
## 
## #####################################################################
## if N:
##     plt.clf()
## 
##     zp = []
##     for i in range(100):
##         for N in range(0,101,10):
##             gd = TradingSyntheticGraph(N=N)
##             zp.append((N, util.evaluate_welfare(gd, algorithm1(gd))))
## 
## 
##     # Extract x and y coordinates from the pairs
##     x_values, y_values = zip(*zp)
##     df = pd.DataFrame({'N': x_values, 'SW': y_values})
##     sns.lineplot(x='N', y='SW', data=df, marker='o', ci='sd', err_style='bars', color='blue', label='Algorithm 1')
## 
##     plt.xlabel('N', fontsize=15)
##     plt.ylabel('Social Welfare', fontsize=15)
##     plt.title('', fontsize=15)
## 
##     # Add grid lines
##     plt.grid(True, linestyle='--', alpha=0.5)
## 
##     # Add a legend with increased font size
##     plt.legend(fontsize=15)
## 
##     # Save the plot as a PDF file
##     plt.savefig('N.pdf', format='pdf')
## 
## #######################################################
## if beta:
##     plt.clf()
## 
##     zp = []
##     for i in range(100):
##         for beta in range(0,11):
##             gd = TradingSyntheticGraph(beta_low=beta/10, beta_high=beta/10)
##             zp.append((beta/10, util.evaluate_welfare(gd, algorithm1(gd))))
## 
## 
##     # Extract x and y coordinates from the pairs
##     x_values, y_values = zip(*zp)
##     df = pd.DataFrame({'beta': x_values, 'SW': y_values})
##     sns.lineplot(x='beta', y='SW', data=df, marker='o', ci='sd', err_style='bars', color='blue', label='Algorithm 1')
## 
##     plt.xlabel('Beta', fontsize=15)
##     plt.ylabel('Social Welfare', fontsize=15)
##     plt.title('', fontsize=15)
## 
##     # Add grid lines
##     plt.grid(True, linestyle='--', alpha=0.5)
## 
##     # Add a legend with increased font size
##     plt.legend(fontsize=15)
## 
##     # Save the plot as a PDF file
##     plt.savefig('beta.pdf', format='pdf')
## 
## 
## #########################################
## 
## if betadiff:
##     plt.clf()
## 
##     zp = []
##     for i in range(100):
##         for beta in range(0,6):
##             gd = TradingSyntheticGraph(beta_low=0.5-beta/10, beta_high=0.5+beta/10)
##             zp.append(("{},{}".format(round(0.5-beta/10,1),round(0.5+beta/10,1)), util.evaluate_welfare(gd, algorithm1(gd))))
## #    2 * beta / 10
## 
##     # Extract x and y coordinates from the pairs
##     x_values, y_values = zip(*zp)
##     df = pd.DataFrame({'beta': x_values, 'SW': y_values})
##     sns.lineplot(x='beta', y='SW', data=df, marker='o', ci='sd', err_style='bars', color='blue', label='Algorithm 1')
## 
##     plt.xlabel('Beta (low,high)', fontsize=15)
##     plt.ylabel('Social Welfare', fontsize=15)
##     plt.title('', fontsize=15)
## 
##     # Add grid lines
##     plt.grid(True, linestyle='--', alpha=0.5)
## 
##     # Add a legend with increased font size
##     plt.legend(fontsize=15)
## 
##     # Save the plot as a PDF file
##     plt.savefig('betadiff.pdf', format='pdf')
## 
## if heatmap:
## 
##     plt.clf()
## 
##     zp = []
##     for i in range(100):
##         for gamma in {0.1, 0.5, 0.9}:
##             for delta in range(0,11):
##                 for beta_high in range(1,11):
##                     gd = TradingSyntheticGraph(gamma=gamma, beta_low=1, beta_high=beta_high, delta=delta/10)
##                     zp.append((beta_high,delta/10,  util.evaluate_welfare(gd, algorithm1(gd)),gamma))
## 
##     # Convert the data to a DataFrame
##     df = pd.DataFrame(zp, columns=[r'$\beta_{\mathrm{high}}/\beta_{\mathrm{low}}$','δ', 'Social Welfare', "Category"])
## 
##     for category, group in df.groupby('Category'):
##         agg_df = group.groupby([r'$\beta_{\mathrm{high}}/\beta_{\mathrm{low}}$','δ'])['Social Welfare'].mean().reset_index()
##         pivot_table = agg_df.pivot(r'$\beta_{\mathrm{high}}/\beta_{\mathrm{low}}$','δ', 'Social Welfare')
## 
##         plt.figure(figsize=(10,10))  # Adjust the figure size as needed
##         hmp = sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=True, yticklabels=True)
## 
##         plt.xlabel("δ", fontsize=20)
##         plt.ylabel(r'$\beta_{\mathrm{high}}/\beta_{\mathrm{low}}$', fontsize=20)
## 
##         # Increase the font size of x and y axis ticks
##         plt.xticks(fontsize=20)
##         plt.yticks(fontsize=20)
## 
## 
## #        plt.tick_params(axis='x', labeltop=True, labelbottom=False)
##         plt.title(f"γ={category}", fontsize=20)
##         pdf_file_name = f'heatmap_{category}.pdf'
##         plt.savefig(pdf_file_name, format='pdf', bbox_inches='tight')
