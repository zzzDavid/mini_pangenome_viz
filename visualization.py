import pickle
from matplotlib import image
import matplotlib.pyplot as plt
import imageio
import numpy as np

with open("./coord_changes.bin", 'rb') as f:
    coord_changes = np.fromfile(f, dtype=np.float32)

coord_changes = coord_changes.reshape((-1, 5, 2))

print(coord_changes)


def draw_line(points, start_idx, end_idx):
    plt.plot([points[start_idx][0], points[end_idx][0]], [points[start_idx][1], points[end_idx][1]], marker='o', linestyle='-')

frames = list()
for idx, frame in enumerate(coord_changes):
    fig, ax = plt.subplots(1)
    ax.set_title("step {}".format(idx))
    # for point_idx in range(5):
    #     x, y = frame[point_idx][0], frame[point_idx][1]
    #     print(f'x={x}, y={y}')
    #     circle = plt.Circle((x, y), 0.05, color='g')
    #     ax.add_patch(circle)

    # point0 -> point1
    draw_line(frame, 0, 1)
    # point1 -> point2
    draw_line(frame, 1, 2)
    # point2 -> point3
    draw_line(frame, 2, 3)
    # point3 -> point4
    draw_line(frame, 3, 4)
    # point2 -> point4
    draw_line(frame, 2, 4)
        

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.savefig(f'figures/{idx}.png') 
    frames.append(imageio.imread(f'figures/{idx}.png'))

imageio.mimsave('animation.gif', frames, duration=0.1)