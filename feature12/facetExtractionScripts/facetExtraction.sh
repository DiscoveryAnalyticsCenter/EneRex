# Syntactic facet based extraction of source code related sentences & dataset related sentences
# Output folder is /home/raquib/SLIE/output/facetBased
# Output files have lots of saved info as this was built for building training data and to provide glimpse into the related sentences' syntactic templates

# creates dataSentences.json and sourceSentences.json files
python facetBasedExtraction.py \
--jsonFilePath data/JSONs/ \
--output output/facetBased


# Findings Links for Source codes, this will create sourceFilteredLink.json file in the output folder
python findingLinks_source.py \
--input output/facetBased \
--output output/facetBased \
--n_core 40 \
--jsonFilesPath data/JSONs


# Findings Links/Refered paper for dataset names, this will create dataFilteredLink.json file in the output folder
python facet_findingLinks.py \
--input output/facetBased \
--output output/facetBased \
--n_core 40 \
--jsonFilesPath data/JSONs/






