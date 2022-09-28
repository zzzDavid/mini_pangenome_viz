import numpy as np

"""
    for n points, the shape of graph_data is
    (path_count, 2 * n, 2 * n)
    Also the graph_data is asymmetric (upper triangular)
"""

def preprocess_graph_data(graph_data):
    """
    Create a symmetric ground truth distance matrix (n, n)
    n is the number of points
    """
    path_count = graph_data.shape[0]
    n_endpoints = graph_data.shape[1]  
    # merge endpoints to find out real number of points
    unique_point = 0
    endp2point = dict()
    dist_matrix = np.zeros((n_endpoints, n_endpoints), dtype=np.int32)
    for i in range(n_endpoints):
        
        if i not in endp2point:
            endp2point[i] = unique_point
            unique_point += 1

        for j in range(i + 1, n_endpoints):
            connections = graph_data[:, i, j]
            if np.count_nonzero(connections) == 0:
                import ipdb; ipdb.set_trace()
                if j not in endp2point:
                    endp2point[j] = unique_point
                    unique_point += 1
                dist = n_endpoints
                dist_matrix[endp2point[i], endp2point[j]] = dist
                dist_matrix[endp2point[j], endp2point[i]] = dist
            elif np.max(connections) < 1e-3:
                # endpoints i and j are the same point
                endp2point[j] = endp2point[i]
            else:
                if j not in endp2point:
                    endp2point[j] = unique_point
                    unique_point += 1
                connections = connections[connections > 0]
                dist = np.min(connections)
                dist_matrix[endp2point[i], endp2point[j]] = dist
                dist_matrix[endp2point[j], endp2point[i]] = dist

    print("there are {} unique points".format(unique_point))
    dist_matrix.tofile("./dist_matrix.bin")

if __name__ == "__main__":
    graph_data = a = np.load('/work/shared/common/datasets/pangenome/distance_npy/DRB1-3123_Dist.npy')
    preprocess_graph_data(graph_data)
    # graph_data = graph_data.astype(np.float32)
    # graph_data.tofile('./preprocess/data.bin')
