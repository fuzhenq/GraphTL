import numpy as np
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from tqdm import tqdm



class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

    def __init__(self):
        super(Graph, self).__init__(list)
    def nodes(self):
        return self.keys()

    def make_consistent(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
        return self
    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return self.order()
    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)


    def random_walk(self, path_length,nb_nod, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.

            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        walk = np.zeros((1, nb_nod))
        walk = np.asarray(walk)
        if start:
            path = [start]
            walk[0][start-1] += 1
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            start = rand.choice(list(G.keys()))
            path = [start]
            walk[0][start-1] += 1

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    next = rand.choice(G[cur])
                    path.append(next)  # G[cur]为顶点cur的邻居节点集合，使用210行的load_edgelist生成图的方式
                    walk[0][next - 1] += 1
                else:
                    path.append(path[0])
                    walk[0][start - 1] += 1
            else:
                break
        return walk


# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length,nb_nod, alpha=0,
                          rand=random.Random(0)):
    walks = np.zeros((nb_nod, nb_nod))
    walks = np.asarray(walks)
    nodes = list(G.nodes())

    for cnt in tqdm(range(num_paths), desc='Walk iteration'):  # 每个节点的path数量
        rand.shuffle(nodes)
        for node in nodes:  # nodes是打乱的节点->keys
            walks[node-1] = walks[node-1] + G.random_walk(path_length, rand=rand, alpha=alpha, start=node,nb_nod=nb_nod)[0]

    return walks


def load_edgelist(adj, directed=False):
    G = Graph()
    # print(adj.shape)

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if(adj[i][j]>0):
                G[i+1].append(j+1)
                # if not directed:
                #     G[j+1].append(i+1)
    G.make_consistent()
    print(G)
    return G
