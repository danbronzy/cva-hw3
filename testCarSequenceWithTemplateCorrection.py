import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=.01, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
epsilon = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = np.array([59, 116, 145, 151])
firstImg = seq[:,:,0]

#containers for the rectangles frame by frame
rects = np.zeros((seq.shape[2], 4))
rects[0,:] = rect

#used to track the difference of the rectangle from the original configuration
diffAccum = np.zeros((1,2))
for frame in range(1, seq.shape[2]):
    print("========= {} =========".format(frame))
    print("diffAccum: {}".format(diffAccum))
    #get the last rectangle and apply standard lucas kanade
    if (frame == 20):
        print("asd")

    lastRect = rects[frame-1,:]

    subDiff = (lastRect - rect)[0:2]
    print("subDiff: {}".format(subDiff))
    p = LucasKanade(seq[:,:,frame-1], seq[:,:,frame], lastRect, threshold, int(num_iters))

    #accumulate template difference
    pDiffTemp = diffAccum + p

    #apply lucas kanade but with the original template. Use pDiff to capture newest template location

    pStar = LucasKanade(firstImg, seq[:,:,frame], rect, threshold, int(num_iters), pDiffTemp)

    dpDiff = np.linalg.norm((pStar - diffAccum) - p)
    print("Diff: {}".format(dpDiff))
    nextRect = np.zeros((1,4))
    #if the pStar parameters don't differ too much
    if dpDiff < epsilon:
        print("Template corrected")
        # Use this set of parameters, simple assignment since pStar includes the original transformation
        diffAccum = pStar
        nextRect = rect + np.hstack((pStar, pStar)).squeeze()
    else:
        # Nah just use the normal ones
        diffAccum = diffAccum + p
        nextRect = lastRect + np.hstack((p, p)).squeeze()

    # Update rect
    rects[frame,:] = nextRect



np.save('../result/carseqrects-wcrt.npy', rects)
oldRects = np.load("../result/carseqrects.npy")
for frameNum in [1, 100, 200, 300, 400]:

    tr = rects[frameNum, :]

    xy = [tr[0], tr[1]]
    width =  tr[2] - tr[0]
    height = tr[3] - tr[1]

    olr = oldRects[frameNum, :]
    oxy = [olr[0], olr[1]]
    owidth =  olr[2] - olr[0]
    oheight = olr[3] - olr[1]

    fig,ax = plt.subplots(1)
    ax.imshow(seq[:,:,frameNum], cmap = 'gray')

    thisRect = patches.Rectangle(xy, width, height, fill=False, edgecolor = [1,0,0])
    oldRect =  patches.Rectangle(oxy, owidth, oheight, fill=False, edgecolor = [0,0,1])
    ax.add_patch(thisRect)
    ax.add_patch(oldRect)

    plt.axis('off')

    plt.savefig("../images/1_4_{}.png".format(frameNum), pad_inches=0, bbox_inches='tight', transparent=True)

