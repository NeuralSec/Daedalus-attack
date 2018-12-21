import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PATHS = ['adv_examples/plot/L2/f3/benign',
         'adv_examples/plot/L2/f3/0.3',
         'adv_examples/plot/L2/f3/0.7']

RESULT_PATH = ['adv_examples/plot/L2/f3/benignresult',
               'adv_examples/plot/L2/f3/0.3result',
               'adv_examples/plot/L2/f3/0.7result']


def plot(paths, result_paths):
    b_imgs = []
    b_r = []
    low_imgs = []
    l_r = []
    high_imgs =[]
    h_r = []
    for (root, dirs, files) in os.walk(paths[0]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                originalimgs = cv2.imread(path) # RGB image
                originalimgs = cv2.cvtColor(originalimgs, cv2.COLOR_BGR2RGB)
                originalimgs = cv2.resize(originalimgs, (416, 416), interpolation=cv2.INTER_CUBIC)
                b_imgs.append(originalimgs)
    for (root, dirs, files) in os.walk(result_paths[0]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                originalimgs = cv2.imread(path) # RGB image
                originalimgs = cv2.cvtColor(originalimgs, cv2.COLOR_BGR2RGB)
                originalimgs = cv2.resize(originalimgs, (416, 416), interpolation=cv2.INTER_CUBIC)
                b_r.append(originalimgs)

    for (root, dirs, files) in os.walk(paths[1]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                lowconfimgs = cv2.imread(path) # RGB image
                lowconfimgs = cv2.cvtColor(lowconfimgs, cv2.COLOR_BGR2RGB)
                low_imgs.append(lowconfimgs)
    for (root, dirs, files) in os.walk(result_paths[1]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                lowconfimgs = cv2.imread(path) # RGB image
                lowconfimgs = cv2.cvtColor(lowconfimgs, cv2.COLOR_BGR2RGB)
                l_r.append(lowconfimgs)

    for (root, dirs, files) in os.walk(paths[2]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                highconfimgs = cv2.imread(path) # RGB image
                highconfimgs = cv2.cvtColor(highconfimgs, cv2.COLOR_BGR2RGB)
                high_imgs.append(highconfimgs)
    for (root, dirs, files) in os.walk(result_paths[2]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                highconfimgs = cv2.imread(path) # RGB image
                highconfimgs = cv2.cvtColor(highconfimgs, cv2.COLOR_BGR2RGB)
                h_r.append(highconfimgs)

    b_imgs = np.array(b_imgs)
    b_r = np.array(b_r)
    low_imgs = np.array(low_imgs)
    l_r = np.array(l_r)
    high_imgs = np.array(high_imgs)
    h_r = np.array(h_r)
    print(b_imgs.shape, b_r.shape, low_imgs.shape, l_r.shape, high_imgs.shape, h_r.shape)
    results = np.stack((b_imgs, b_r, low_imgs, l_r, high_imgs, h_r))

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(6, 10, wspace=0.1, hspace=0.1)

    for i in range(6):
        for j in range(10):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(results[i, j], interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
    gs.tight_layout(fig)
    plt.show()

if __name__ == '__main__':
    plot(PATHS, RESULT_PATH)
