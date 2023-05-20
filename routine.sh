# compute descriptors from a dataset
python3 --dataset ../data/VGG-Face2/data/test/n000001 --descriptor SIFT --output output_descriptor/descriptor

# construct a visual dictionary from the descriptors
python3 visualdictionary.py -d output_descriptor/descriptor.pickle -w 16 -o output_visual/visual

# Compute VLAD descriptors from the visual dictionary
python3 vladdescriptors.py --dataset ../data/VGG-Face2/data/test/n000001 --visualdictionary output_visual/visual.pickle --descriptor SIFT -o output_vlad/vladh
