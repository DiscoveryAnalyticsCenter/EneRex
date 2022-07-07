# Dataset and Source code feature extractions

## Environment and Dependencies
Please install the provided `environment.yml` or `environment_minimal.yml` file to get the conda environment necessary to run the codes in this subtask.

If `environment.yml` fails, please try with `environment_minimal.yml`. With this one, you will need to install spacy(2.2.4), seqeval and pytorch separately. Please use `pip install spacy==2.2.4`, `pip install seqeval` and follow [torch website](https://pytorch.org/get-started/locally/) to install compatible version of torchvision. **Addition: Please also install `fuzzywuzzy[speedup]` and `networkx` using pip after setting up the environment**.

**In both cases, Please use `python -m spacy download en_core_web_sm` to download the spacy english model. You will also need to install the huggingface transformer by runing `pip install .` inside the `transformers` directory.** This is the snapshot of huggingface transformer model that we used to develop the codes. Compatibiliy with the future versions will be reviewed in future.


## Training
Training data for the EneRex comes after a facet based extraction process and preprocessing the data. The intermediate training data is placed inside `noncontext` directory for both source code and dataset. The facet based extraction codes are inside `facetExtractionScripts`. if you want to run the data starting with a sample set of PDF files by yourself. Please check the `train_script.sh` for training a new model for both facets, one at a time.


## Prediction
Transformer based extraction(source code and dataset) is done by `transformerExtraction.sh` script inside `predictionScript` directory. It will work with the JSON files in `data/JSONs/` directory. It consist of a sentence classifier, followed by a NER model for finding entities and two python scripts for finding Links/Refered paper for source code and dataset sentences. The necessary steps are described below:

1. This script requires three models. The first two of them are sentence classifiers, one for dataset feature and another for souce code feature. The third model is the Named entity recognizer for dataset features. After training the models, please update the paths names in the scripts. Please see below for more details.

    a. Dataset Classifier (Transformer sentence classifier fine tuned on Sci-BERT)  
    b. Source Code Classifier (Transformer sentence classifierfine tuned on Sci-BERT)  
    c. Named Entity Recognizer (Transformer NER fine tuned on Sci-BERT)  

2. The path to the first two sentence classifiers are in `modelLocation.json` file that the sentence classifier algo will use. It also needs a temporary data directory that will be used to save temporary data. **Please make directory `tempData` in this directory(`feature12/`)**. If you change the first two model's path, please update the `modelLocation.json` file. Otherwise, proceed with the default path values. 

3. For the NER, the script will call `NERmodel/predict_NERScript.sh`. You can check that script to tweak options for the NER model. Again, you will need to update the NER model's path here after train the model.

4. Run the `transformerExtraction.sh`. **The `CUDA_VISIBLE_DEVICES` in the scripts  is defaulted to `0` but change as necessary**. `transformerExtraction.sh` calls `NERmodel/predict_NERScript.sh` to take care of NER task. **You will need to update `CUDA_VISIBLE_DEVICES` there too if you would like to change it from default `0`**

5. The outputs of the transformer based algorithm are in `output/transformer` directory. `dataSentences.json` files contains dataset related sentences for each papers. Each key is an Arxiv id. It also has previous sentence, next sentence, the spans of entities present. `sourceSentences.json` is also in similar format minus the entity/spans. `dataNNResult.json` file contains the compiled dataset names for each paper and each key is an Arxiv id. `sourceFilteredLink.json` contains any URL related to the source code. `dataFilteredLink.json` is the same thing for dataset names.