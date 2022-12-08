'''
Labeling functions for RNA dataset.
'''

# Labeling functions.

@labeling_function()
def exclude_animals(x):
    return EXCLUDE if any(animal in x.Title for animal in unwanted_animals) else ABSTAIN

@labeling_function()
def exclude_terms(x):
    return EXCLUDE if any(term in x.Title for term in unwanted_terms) else ABSTAIN

@labeling_function()
def include_necessary_terms(x):
    return INCLUDE if any(term in x.Title for term in necessary_terms) else EXCLUDE

# Data for labeling functions.
unwanted_animals = ["xenopus", "marine", "aquatic", "cow", "Bos", "bovine", "cattle", "sheep",
                    "ovine", "caprine", "goat", "livestock", "pig", "porcine", "amphibian",
                    "reptile", "frog", "snake", "fish", "cichlid", "crustacean", "crab", "insect",
                    "chicken", "duck", "mallard", "finch", "bird", "avian", "arachnid", "yak",
                    "chordate", "Ciona", "trout", "salmon", "heifer"]
unwanted_terms = ["NMR", "MD", "molecular dynamic", "RNA world", "wastewater",
                  "water", "viral load", "RNA degradation", "RNA structur", "secondary structur",
                  "folding", "regulat", "promoter", "spectroscop", "spectr", "RNA decay", "RNA replication",
                  "transcription factor", "switch", "control", "RNA metaboli",
                  "population dynamics", "evolution", "protein", "conformation", "viral dynamics",
                  "polymerase", "telomerase RNA", "Structural dynamics", "control", "ribosom",
                  "repuckering dynamics", "replication system", "loop", "hairpin", "chromatin dynamics",
                  "dynamic modularity", "imaging", "reporter", "kinetic",
                  "developmental dynamics", "temporal lobe", "epilep",
                  "cytometry", "imag", "fluoresc", "telomer", "composition dynamics", "noncoding",
                  "non-coding", "buckl", "spin", "energy", "RNA quality"]
necessary_terms = ["rna-seq", "microarray", "gene expression", "transcriptom"]
