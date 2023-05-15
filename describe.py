# compute SIFT, SURF or ORB descriptors from an image dataset
# USAGE :
# python describe.py --dataset dataset --descriptor descriptorName --output output
# or 
# python describe.py -d dataset -n descriptorName -o output
# example :
# python describe.py --dataset dataset --descriptor SURF --output descriptorSURF
# python describe.py --dataset dataset --descriptor SIFT --output descriptorSIFT
# python describe.py --dataset dataset --descriptor ORB --output descriptorORB

from src.descriptors import describeSIFT
from src.descriptors import describeSURF
from src.descriptors import describeORB
import argparse
import glob
import cv2
import itertools
import numpy as np
import pickle

# parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images")
ap.add_argument("-n", "--descriptor", required=True, help="descriptor = SURF, SIFT or  ORB")
ap.add_argument("-o", "--output", required=True, help="Path to where the computed descriptors will be stored")
args = vars(ap.parse_args())


def get_descriptors(path, functionHandleDescriptor):
    descriptors = []

    for imagePath in glob.glob(path + "/*.jpg"):
        im = cv2.imread(imagePath)
        keypoints, desc = functionHandleDescriptor(im)
        if descriptors is not None:
            descriptors.append(desc)

        print('Num. descriptors: {}, image: {}.'.format(len(keypoints), imagePath))
    # flatten list
    descriptors = list(itertools.chain.from_iterable(descriptors))
    # list to array
    descriptors = np.asarray(descriptors)

    return descriptors


# reading arguments
path = args["dataset"]
descriptorName = args["descriptor"]
output = args["output"]

# computing the descriptors
handler_dict = {"SURF": describeSURF, "SIFT": describeSIFT, "ORB": describeORB}
descriptors = get_descriptors(path, handler_dict[descriptorName])

# writting the output
file = output + ".pickle"

with open(file, 'wb') as f:
    pickle.dump(descriptors, f)
