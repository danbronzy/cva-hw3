import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.01, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

rects = np.zeros((seq.shape[2], 4))
rects[0,:] = rect
for frame in range(1, seq.shape[2]):
    lastRect = rects[frame-1,:]
    p = LucasKanade(seq[:,:,frame-1], seq[:,:,frame], lastRect, threshold, int(num_iters))

    # Update rect
    nextRect = lastRect + np.hstack((p, p)).squeeze()
    rects[frame,:] = nextRect

np.save('../result/carseqrects.npy', rects)

for frameNum in [1, 100, 200, 300, 400]:
    tr = rects[frameNum, :]

    xy = [tr[0], tr[1]]
    width =  tr[2] - tr[0]
    height = tr[3] - tr[1]

    fig,ax = plt.subplots(1)
    ax.imshow(seq[:,:,frameNum], cmap = 'gray')

    thisRect = patches.Rectangle(xy, width, height, fill=False, edgecolor = [1,0,0])
    ax.add_patch(thisRect)
    plt.axis('off')

    plt.savefig("../images/1_3_{}.png".format(frameNum), pad_inches=0, bbox_inches='tight', transparent=True)

