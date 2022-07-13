# EneRex: End-to-end Research Entity Extractor 
This is the public repository with the code and data from the paper ["Lessons from Deep Learning applied to Scholarly Information Extraction: What Works, What Doesn’t, and Future Directions"](https://api.semanticscholar.org/CorpusID:250408178)

## Structure and Overview of the data directories
The different facets of EneRex is gropued into three pipeline, Source code and dataset are inside `feature12`. Objective task and methods are inside `feature345`. Hardware and language/library are inside `hardware_language_library_extractor`. Each directory has own readme file with details. 

We have a `data` directory here that holds input files(pdf,txt) and their structured representation that will be used throughout the `Task1` pipeline. There are two ways to produce input(for both train and predict) for our system. You can use PDF and TXT. **Directory `data/JSONs/` contains structured representation of the papers that is used as input for our extraction scripts down the line. This is the single data entry point for all extraction subtasks.**


### 1. Working with PDF files
Steps to run GROBID server and Parser are given below:

1. [Visit this link](https://grobid.readthedocs.io/en/latest/Install-Grobid/) to download and install Grobid. Make sure Java in installed. After installation, build the gradle by following documentation and run the service. The usual commands are these: `cd grobid-0.5.6` and build with `./gradlew clean install`. Run with `./gradlew run`. Please note that Grobid 0.5.6 is tested with the codes. You can install latest versions but compatibility is not checked at this moment(will be added later). 
2. The GROBID runs as web API. We are using a python client. You need beautifulsoup4(with lxml) and requests packages for this script. You can make a env with `conda create --name grobid bs4 requests lxml`. Place PDF file in `data/PDFs/` and run `Grobid/grobidParser.sh` script to get the output in `data/JSONs/` directory. `data/TEIXMLs/` is just a temporary directory to hold XML files from Grobid output. We also placed file sample PDF files in the `data/PDFs/` that you can test for 'working with PDF' part.


### 2. Working with TXT files
Please place TXT files in `data/TXTs/` to convert TXT files into the structured input format the extraction pipeline requires. Files should have only two lines, First one with title and second one with abstract. We developed the codes to utilize the section and structural information for each paper. So when there are only title and abstracts, the goal is to create that compatible JSON files that our subsequent codes can use. Run `Grobid/TXTtoJSON.py` in following format:

`python TXTtoJSON.py --input ../data/TXTs/ --output ../data/JSONs/`

## Format of `data/JSONs/` files
Each JSON file represents a single PDF/TXT file. It contains a single list. There are five dictionaries inside this list. The first one has some metadata about the paper. Second dictionary has sections' name and text. Third one has the footnotes, fourth one has the references and fifth one has the tables. In case of txt files, only first two dictionaries are populated and rest three are left empty.


## Features list
Directory `feature12` has the source code and dataset feature extraction scripts. Direcotry `feature345` has task, application and method feature extraction scripts. Please review the `README` for each subtask/feature directory for environment and dependencies.

## Citation
If you find this project useful for your task/research please cite using the following bibtex file.

    @inproceedings{yousuf2022lessons,
    title={Lessons from Deep Learning applied to Scholarly Information Extraction: What Works, What Doesn't, and Future Directions},
    author={Yousuf*, Raquib Bin and Biswas*, Subhodip and Kaushal, Kulendra Kumar and Dunham, James and Gelles, Rebecca and Muthiah, Sathappan and Self, Nathan and Butler, Patrick and Ramakrishnan, Naren},
    booktitle={Proceedings of the 28th ACM SIGKDD international conference on Knowledge discovery and data mining},
    series={Workshop on Data-Driven Science Of Science},
    year={2022},
    doi = {10.48550/ARXIV.2207.04029},
    url = {https://arxiv.org/abs/2207.04029},
    }



<!-- If path is not set, set path as `export PATH=/home/raquib/jdk1.8.0_131/bin:$PATH`.  -->