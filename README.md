# TASK1: Extraction of Features

## Structure and Overview of the data directories
This is the root directory of **Task1**. We have a `data` directory here that holds input files(pdf,txt) and their structured representation that will be used throughout the `Task1` pipeline. There are three ways to produce input for our system. You can use PDF, TXT and CSET annotated jsonl files. **Directory `data/JSONs/` contains structured representation of the papers that is used as input for our extraction scripts down the line. This is the single data entry point for all extraction subtasks.**


### 1. Working with PDF files
Steps to run GROBID server and Parser are given below:

1. [Visit this link](https://grobid.readthedocs.io/en/latest/Install-Grobid/) to download and install Grobid. Make sure Java in installed. After installation, build the gradle by following documentation and run the service. The usual commands are these: `cd grobid-0.5.6` and build with `./gradlew clean install`. Run with `./gradlew run`. Please note that Grobid 0.5.6 is tested with the codes. You can install latest versions but compatibility is not checked at this moment(will be added later). 
2. The GROBID runs as web API. We are using a python client. You need beautifulsoup4(with lxml) and requests packages for this script. You can make a env with `conda create --name grobid bs4 requests lxml`. Place PDF file in `data/PDFs/` and run `Grobid/grobidParser.sh` script to get the output in `data/JSONs/` directory. `data/TEIXMLs/` is just a temporary directory to hold XML files from Grobid output. 


### 2. Working with TXT files
Please place TXT files in `data/TXTs/` to convert TXT files into the structured input format the extraction pipeline requires. Files should have only two lines, First one with title and second one with abstract. We developed the codes to utilize the section and structural information for each paper. So when there are only title and abstracts, the goal is to create that compatible JSON files that our subsequent codes can use. Run `Grobid/TXTtoJSON.py` in following format:

`python TXTtoJSON.py --input ../data/TXTs/ --output ../data/JSONs/`

### 3. Working with CSET annotated jsonl files
These are the task-method annotated jsonl files from the CSET. We are only using the title and abstract part here. Please place jsonl files in `data/CSETFormat/` to convert jsonl files into the structured input format the extraction pipeline requires. We developed the codes to utilize the section and structural information for each paper. As there are only title and abstract inside these jsonl files, the goal is to create that compatible JSON files that our subsequent codes can use. Run `Grobid/CSETtoJSON.py` in following format:

`python CSETtoJSON.py --input ../data/CSETFormat/ --output ../data/JSONs/`

Please note that data directories now contain input and output files of five sample papers of CSET annotated jsonl files. We also placed file sample PDF files in the `data/PDFs/` that you can test for 'working with PDF' part.

## Format of `data/JSONs/` files
Each JSON file represents a single PDF/TXT file. It contains a single list. There are five dictionaries inside this list. The first one has some metadata about the paper. Second dictionary has sections' name and text. Third one has the footnotes, fourth one has the references and fifth one has the tables. In case of txt files and CSET annotated jsonl files, only first two dictionaries are populated and rest three are left empty.


## Features list
Directory `feature12` has the source code and dataset feature extraction scripts. Direcotry `feature345` has task, application and method feature extraction scripts. Please review the `README` for each subtask/feature directory for environment and dependencies.



<!-- If path is not set, set path as `export PATH=/home/raquib/jdk1.8.0_131/bin:$PATH`.  -->