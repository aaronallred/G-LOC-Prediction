import numpy as np

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

#def kNN_classifier(gloc_data)


