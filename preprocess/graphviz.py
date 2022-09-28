# First networkx library is imported
# along with matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Defining a Class
class GraphVisualization:

	def __init__(self):
		
		# visual is a list which stores all
		# the set of edges that constitutes a
		# graph
		self.visual = []
		
	# addEdge function inputs the vertices of an
	# edge and appends it to the visual list
	def addEdge(self, a, b, weight=1):
		# temp = [a, b, weight]
		temp = [a, b]
		self.visual.append(temp)
		
	# In visualize function G is an object of
	# class Graph given by networkx G.add_edges_from(visual)
	# creates a graph with a given list
	# nx.draw_networkx(G) - plots the graph
	# plt.show() - displays the graph
	def visualize(self, filename):
		G = nx.Graph()
		# G.add_weighted_edges_from(self.visual)
		G.add_edges_from(self.visual)

		# Spring layout
		# pos = nx.spring_layout(G)
		# looks messy

		# Force directed layout
		# pos = nx.kamada_kawai_layout(G)
		# very slow
		# but the result looks very promising
		
		# Spectral layout
		pos = nx.spectral_layout(G)
		# looks good
		# pos is good for initialization
		init = np.asarray([v for v in pos.values()]).astype(np.float32)
		init.tofile("./spectral_init.bin")
		
		# Bipartite layout
		# top = nx.bipartite.sets(G)[0]
		# pos = nx.bipartite_layout(G, top)
		# Graph not bipartite, doesn't work
		
		nx.draw_networkx(G, pos=pos, linewidths=0.1, node_size=1, with_labels=False)
		
		plt.savefig(filename, dpi=1000)

if __name__ == "__main__":
	print("reading data...")
	dist_matrx = np.fromfile('./weighted_edge.bin', dtype=np.int32)
	n = int(np.sqrt(dist_matrx.shape[0]))
	dist_matrx = dist_matrx.reshape((n, n))
	G = GraphVisualization()
	nonzeros_idx = np.nonzero(dist_matrx)
	print("creating graph...")
	for i, j in zip(nonzeros_idx[0], nonzeros_idx[1]):
		G.addEdge(i, j, dist_matrx[i, j])
	print("visualizing graph...")
	G.visualize("./spectral_layout.png")
	print("done")