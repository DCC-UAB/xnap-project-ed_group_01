from build_phoc import phoc
import numpy as np

annotations_file = "Datasets/lexicon.txt"

with open(annotations_file, "r") as file:
    list_of_words = file.readlines()

list_of_words = [l[:-1] for l in list_of_words]
phoc_representations = phoc(list_of_words)
suma = np.sum(phoc_representations, axis=0)
weights = phoc_representations.shape[0]/(suma+1e-6)
print("p")
