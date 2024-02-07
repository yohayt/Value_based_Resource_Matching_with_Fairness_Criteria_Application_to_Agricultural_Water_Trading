import random
from pdb import set_trace
from abc import ABC, abstractmethod
import pyarrow
import networkx as nx
epsilon = 0.000001

class GraphData(ABC):
    @abstractmethod
    def f_s(self, i):
        pass

    @abstractmethod
    def f_b(self, i):
        pass

    @abstractmethod
    def get_compatibility_graph(self):
        pass

    @abstractmethod
    def get_rho(self, i):
        pass


class TradingSyntheticGraph(GraphData):
    def __init__(self, N=10, k=5, beta_high=0.7, beta_low=0.3, gamma=0.5, delta=0.5):
        self.N = N  # number of agents
        self.model = "Single Stream"
        self.delta = delta  # drought severity param (0,1)
        self.beta_high = beta_high  # value function factors.
        self.beta_low = beta_low
        self.high = set()
        self.k = k
        self.gamma = gamma

        for i in range(1, self.N + 1):
            if random.random() < self.ph(i):
                self.high.add(i)
        self.compatibility_graph = nx.Graph()

        for i in range(1, self.N + 1):
            if self.is_seller(i):
                self.compatibility_graph.add_node(i, bipartite=1)
            else:
                self.compatibility_graph.add_node(i, bipartite=0)

        buyers = [node for node, attributes in self.compatibility_graph.nodes(data=True) if
                  attributes.get('bipartite') == 0]
        sellers = [node for node, attributes in self.compatibility_graph.nodes(data=True) if
                   attributes.get('bipartite') == 1]

        for buyer in buyers:
            for seller in sellers:
                self.compatibility_graph.add_edge(buyer, seller)  # allow all edges

    def get_compatibility_graph(self):
        return self.compatibility_graph

    def is_seller(self, i):  # agent is seller according to its position
        return (i+epsilon) / self.N >= 1 - self.delta

    def ph(self, i):
        return self.gamma * i / self.N + (1 - self.gamma) * (1 - i / self.N)

    def get_beta(self, i):
        if i in self.high:
            return self.beta_high
        else:
            return self.beta_low

    def f_s(self, i, l):
        return self.get_beta(i) * l

    def f_b(self, i, l):
        return self.get_beta(i) * (self.k - l + 1)

    def get_rho(self, i):
        return self.k
