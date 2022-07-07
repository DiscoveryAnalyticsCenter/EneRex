# Task and method extraction

## Dependencies
This code was developed using Python 3.7. To create a new Conda environment using Python 3.7, do `conda create --name dygiepp python=3.7`.

The necessary dependencies can be installed with `pip install -r requirements.txt`.

The only dependencies for the modeling code are [AllenNLP](https://allennlp.org/) 0.9.0 and [PyTorch](https://pytorch.org/) 1.2.0. It may run with newer versions, but this is not guarenteed. For PyTorch GPU support, follow the instructions on the [PyTorch](https://pytorch.org/).

For data preprocessing a few additional data and string processing libraries are required including, [Pandas](https://pandas.pydata.org) [Beautiful Soup 4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), [scispacy](scispacy) and rank-BM25. These are already included in the requirement file.


## Details
Our algorithm to extract task, application and method terms uses dygie++ system to extract entities and their relations. We then use rules and filters to decide on the task, application and method terms for each paper. Please run the `get_scibert.py` first to get Sci_BERT vocabulary that is necessary for the model.

<!-- - [ ] We are using the SciERC_lightweight model in dygiee++ system for running dygie++ on our data. The model file is already placed in the directory named `pretrained`.  -->

### Input file format
The input to the scripts in this directory is the structured data files (`task1/data/JSONs/`) that our extraction pipeline normally use. Refer to [`task1` directory readme file](https://github.com/DiscoveryAnalyticsCenter/csetproject/tree/master/task1) to view available options to generate structured JSON files. We have three data converter implemented at this moment. a)PDF, b)TXT and c)CSET annotated jsonl files.

Please note that data directories here now contain pipeline outputs for five sample papers of CSET annotated jsonl files. We are also keeping intermediate output files for demonstration purpose. The results are on the following sections of the papers: 'title', 'abstract'. If you use `data/PDFs/` instead, the result of this subtask will be on the following sections of the paper: 'title', 'abstract', 'introduction', 'conclusion', 'discussion' & 'concluding remark'.

### Scripts Details
Please run the `get_scibert.py` first to get Sci_BERT vocabulary that is necessary for the model. Then run one of these two scripts here, single mode and batch mode. 

1. `single_predict.sh` can be used to run small number of papers( prefer<500) in a single run. Directory `outputSingleMode` will be used to store outputs of this mode.

2. `batchPredict.sh` can be used for batch mode. Directory `outputBatchMode` will be used to store outputs of this mode.


### Output details
Directory `outputSingleMode` contains output of single mode and directory `outputBatchMode` contains that of batch mode. In both single or batch modes, `task_app_method.json` and `sentences_task_app_method.json` files are the standard final output. Each key is an Arxiv id/paper file name/doc id. Values are dictionaries with keys `task_application` and `method`. Those directories also holds other intermediate output files. 
