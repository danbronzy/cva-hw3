import argparse
import numpy as np
import matplotlib.pyplot as plt
from SubtractDominantMotion import SubtractDominantMotion

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.082, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')
masks = np.zeros((seq.shape[0], seq.shape[1], seq.shape[2] - 1))
for frame in range(2, seq.shape[2]):
# for frame in [30, 60, 90, 120]:
    print("Frame: {}".format(frame))
    image1 = seq[:,:,frame - 1]
    image2 = seq[:,:,frame]

    mask = SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance)
    masks[:,:,frame - 1] = mask

for frame in [30, 60, 90, 120]:
    image1 = seq[:,:,frame - 1]
    image2 = seq[:,:,frame]

    mask = masks[:,:,frame - 1]

    image1_c = np.dstack((image1, image1))
    blue = np.where(mask, 1.0, image1)
    highlighted = np.dstack((image1_c, blue))

    fig,ax = plt.subplots(1)
    ax.imshow(highlighted)
    plt.axis('off')

    plt.savefig("../images/2_3_aerial_{}.png".format(frame), pad_inches=0, bbox_inches='tight', transparent=True)

    plt.imshow(highlighted)
    plt.show()