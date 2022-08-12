# Published Dataset

We have published two datasets. One is Brat annoated in-house evaluation set we used to measure performance of our model. Another is the Named entity recognizers training data for the computing resourced, language/library (CORLL). Each has corresponding directory.

## Brat Annoated in-house Eval Set
The annotated files (txt and ann) and the config files for BRAT are in `BRAT_annotated_eval/data` directory. `makeList_brat.py` contain the script of making txt files from JSON file of our system. To allow for word and sentence level annotation, we have created 8 entities in BRAT, a word entity and a sentence entity for each of the four feature:

1. Datasets
2. Source Code
3. Computing Resources (hw and time)
4. Language & Library


## CORLL Dataset
CORLL is a sentence level dataset. It is comprised of sentences containing entities that have been used in the article and the text span in which the entity is present in the BILUO format. The dataset contains over 600 such salient sentences and around 1400 annotated entities overall. Furthermore, the computing resources facet is divided into computing platform, compute time, and hardware resources entities. This dataset is divided into three parts, train, dev and test sets.


