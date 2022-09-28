import networkx as nx
import numpy as np


def generate_gt(dist_matrix):
    G = nx.Graph()
    n = dist_matrix.shape[0]
    nonzeros_idx = np.nonzero(dist_matrix)
    print("creating graph...")
    weight_sum = 0
    for i, j in zip(nonzeros_idx[0], nonzeros_idx[1]):
        G.add_edge(i, j, weight=dist_matrix[i, j])
        weight_sum += dist_matrix[i, j]
    gt = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            if G.has_node(i) and G.has_node(j) and nx.has_path(G, i, j):
                path = nx.dijkstra_path_length(G, i, j)
                gt[i, j] = path
                gt[j, i] = path
            else:
                gt[i, j] = weight_sum
                gt[j, i] = weight_sum
    return gt

if __name__ == "__main__":
    print("reading data...")
    dist_matrx = np.fromfile('./weighted_edge.bin', dtype=np.int32)
    n = int(np.sqrt(dist_matrx.shape[0]))
    dist_matrx = dist_matrx.reshape((n, n))
    gt = generate_gt(dist_matrx)
    print("saving ground truth to {}".format("./ground_truth.bin"))
    gt.tofile("./ground_truth.bin")
    print("done")