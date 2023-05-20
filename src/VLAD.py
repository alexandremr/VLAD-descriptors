from sklearn.cluster import KMeans
import glob
import cv2
import numpy as np


def kmeans_dictionary(training, k):
    # K-means algorithm
    est = KMeans(n_clusters=k, init='k-means++', tol=0.0001, verbose=1).fit(training)
    # centers = est.cluster_centers_
    # labels = est.labels_
    # est.predict(X)
    return est
    # clf2 = pickle.loads(s)


def get_VLAD_descriptors(path, handler_descriptor, visual_dictionary):
    descriptors = list()
    idImage = list()
    for image_path in glob.glob(path + "/*.jpg"):
        print(image_path)
        im = cv2.imread(image_path)
        keypoints, desc = handler_descriptor(im)
        if desc is not None:
            v = VLAD(desc, visual_dictionary)
            descriptors.append(v)
            idImage.append(image_path)

    # list to array
    descriptors = np.asarray(descriptors)
    return descriptors, idImage


# fget a VLAD descriptor for a particular image
# input: X = descriptors of an image (M x D matrix)
# visualDictionary = precomputed visual dictionary
def VLAD(X, visual_dictionary):
    predicted_labels = visual_dictionary.predict(X)
    centers = visual_dictionary.cluster_centers_
    labels = visual_dictionary.labels_
    k = visual_dictionary.n_clusters

    m, d = X.shape
    V = np.zeros([k, d])

    for i in range(k):
        if np.sum(predicted_labels == i) > 0:
            V[i] = np.sum(X[predicted_labels == i, :] - centers[i], axis=0)

    V = V.flatten()
    V = np.sign(V) * np.sqrt(np.abs(V))

    # L2 normalization
    V = V / np.sqrt(np.dot(V, V))
    return V
