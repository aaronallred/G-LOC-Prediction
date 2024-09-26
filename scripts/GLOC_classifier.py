import numpy as np
from pygam import GAM, s, f
from pygam.datasets import wage
import matplotlib.pyplot as plt

def categorize_gloc(gloc_data):
    # Create GLOC Classifier Vector
    event = gloc_data['event'].to_numpy()
    event_validated = gloc_data['event_validated'].to_numpy()
    gloc_classifier = np.zeros(event.shape)

    gloc_indices = np.argwhere(event == 'GLOC')
    rtc_indices = np.argwhere(event_validated == 'return to consciousness')

    for i in range(gloc_indices.shape[0]):
        gloc_classifier[gloc_indices[i, 0]:rtc_indices[i, 0]] = 1

    return gloc_classifier


def gam_classifier(X,y):

    ## model
    gam = GAM(s(0, n_splines=5), distribution='binomial', link='logit')
    gam.fit(X, y)

    ## plotting
    plt.figure();
    fig, axs = plt.subplots(1, 1);

    titles = ['HR 95% Conf Bound']

    ax = axs
    i=0
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    plt.scatter(X, y, facecolor='gray', edgecolors='none')
    ax.set_title(titles[i]);

    plt.show()

    gam.summary()