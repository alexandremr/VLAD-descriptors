from src.VLAD import kMeansDictionary
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--descriptors_path", required=True, help="Path to the file that contains the descriptors")
ap.add_argument("-w", "--number_visual_words", required=True, help="number of visual words or clusters to be computed")
ap.add_argument("-o", "--output_descriptor", required=True, help="Path to where the computed visualDictionary will be stored")
args = vars(ap.parse_args())

path = args["descriptors_path"]
k = int(args["number_visual_words"])
output = args["output_descriptor"]

print("estimating a visual dictionary of size: " + str(k) + " for descriptors in path:" + path)

with open(path, 'rb') as f:
    descriptors = pickle.load(f)

visualDictionary = kMeansDictionary(descriptors, k)

# output_descriptor
file = output + ".pickle"

with open(file, 'wb') as f:
    pickle.dump(visualDictionary, f)

print("The visual dictionary  is saved in " + file)
