from build_phoc import phoc

annotations_file = "Datasets/lexicon.txt"

with open(annotations_file, "r") as file:
    list_of_words = file.readlines()

list_of_words = [l[:-1] for l in list_of_words]
phoc_representations = phoc(list_of_words)
print("p")
