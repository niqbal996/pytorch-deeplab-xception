'''
To run the TSNE from cuml you need to create a virtual environment according to your PC configuration.
visit the URL (https://rapids.ai/start.html#rapids-release-selector)
e.g.

conda create -n rapids-0.18 -c rapidsai-nightly -c nvidia -c conda-forge \
    -c defaults blazingsql=0.18 cuml=0.18 python=3.7 cudatoolkit=11.0
'''

import numpy as np
import matplotlib.pyplot as plt
from cuml.manifold import TSNE
from glob import glob

files = glob('../feature_maps/cat_1/*.npy')

for file, i in zip(files, range(len(files))):
    print('[INFO] Processing file {}/{} . . .'.format(i+1, len(files)))
    with open(file, 'rb') as f:
        x = np.load(f)

    x = np.transpose(x)

    tsne = TSNE(n_components=2)
    X_hat = tsne.fit_transform(x)

    plt.scatter(X_hat[:, 0], X_hat[:, 1])

plt.show()