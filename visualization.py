import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embeddings(embeddings, label_file):

    emb_list = embeddings
    Y = label_file

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(Y)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)

    plt.legend()
    plt.show()