import pickle
from matplotlib import image
import matplotlib.pyplot as plt
import imageio
import numpy as np

with open("./coord_changes.bin", 'rb') as f:
    coord_changes = np.fromfile(f, dtype=np.float32)

npoint = 3521
coord_changes = coord_changes.reshape((-1, npoint, 2))

# print(coord_changes)
xmin = np.min(coord_changes, axis=(0, 1))[0]
xmax = np.max(coord_changes, axis=(0, 1))[0]
ymin = np.min(coord_changes, axis=(0, 1))[1]
ymax = np.max(coord_changes, axis=(0, 1))[1]


def draw_line(points, start_idx, end_idx):
    plt.plot([points[start_idx][0], points[end_idx][0]], [points[start_idx][1], points[end_idx][1]], marker='o', linestyle='None', markersize=1)

frames = list()
for idx, frame in enumerate(coord_changes):
    fig, ax = plt.subplots(1)
    ax.set_title("step {}".format(idx))
    for point_idx in range(npoint-1):
        draw_line(frame, point_idx, point_idx + 1)
    # for point_idx in range(npoint):
    #     ax.scatter(frame[0], frame[1], marker='o', alpha=0.6, color='#17becf')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.savefig(f'figures/{idx}.png') 
    frames.append(imageio.imread(f'figures/{idx}.png'))

imageio.mimsave('animation.gif', frames, duration=0.1)