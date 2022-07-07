#CSET Hardware, Language and Library extraction Source Code

# <p align=center>`HLL Extractor`</p>
HLL Extractor has been trained on scholarly articles from the arXiv corpus.

* HLL Extractor has been developed with an aim of extracting Technical Details from Scholarly Literature.
* HLL Extractor takes research paper in JSON format after being processesd from GROBID. 
* Given a research paper(s) in JSON, HLL extracts theto find the following items.
   * Computational platform utilized in the paper, 
   * Language/library dependencies, and
   * Compute time and resources consumed.
   
<!-- ### Downloading Trained Models

* `[Sentence Classification Model]`
    * Hardware Sentence Classifier
    * Language Sentence Classifier
    * Library Sentence Classifier
* `[CLustering Model]`
    * Hardware Clustering Model
    * Language Clustering Model
    * Library Clustering Model
* `[NER Model]` 
    * HLL NER Model
    * HLL NER Model identifies following entities from a sentence. 
        * Hardware Platform
        * Hardware Resources
        * Compute Time
        * Programming Language
        * Programming Library -->
        
### Using HLL Extractor

Complete the following prerequisites:

* Setup the Python 3.6 virtual environment.
* Install the dependencies using `pip install -r requirements.txt`

* For using prediction pipeline:
    * Choose one yml file inside `prediction_pipeline/config` and update the following variables in that file. 
    * Point `MODELS_FOLDER_BASE_PATH` in `prediction_pipeline/config` to the models folder present inside the repository `models`.
    * Point `LIST_OF_PDFs` in `prediction_pipeline/config` to a text file having name of all the research papers one on a line, a sample file has been included in the `sample_files/filelist.txt`.
    * Point `OUTPUT_FOLDER` in `prediction_pipeline/config` to the folder where you want to generate outputs (can create an empty folder and point to it).
    * Point `PDF_FOLDER_PATH` in `prediction_pipeline/config` to the folder containing all research paper in JSON, sample files has been included in the `sample_files/pdf_folder`.
    * `prediction_piline/extraction_script` is the script to run prediction pipeline. Mention 
    the yml file as environment variable inside this script and run that. This will put output of all the hardware related features in files for each paper
    * The post-processing filtering is used to improve precision on the predicted outputs. `prediction_piline/post_process_library.py` is for library_language and `prediction_piline/post_process_computing.py` is for computing resources. Each of the script take the input and output as argument. Input is the directory where previous `prediction_piline/extraction_script` outputs all the results. Please provide a new output location for that certain feature.
    
* For using training pipeline :
    * Point the `TRAINING_DATA_BASE_PATH` in `training_pipeline/config` to the folder which will contain training data.
    * Point the `MODELS_FOLDER_BASE_PATH` in `training_pipeline/config` to the folder containing all the pretrained models.
    * Point the `OUTPUT_FOLDER_BASE_PATH` in `training_pipeline/config` to the folder which will contain the training outputs.
    
    * For training Sentence Classifier
        * Put all the hardware positive class sentences in a text file named `positive_hardware_sents.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`.
        * Put all the hardware negative class sentences in a text file named `negative_hardware_sents.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`..
        * Put all the language positive class sentences in a text file named `positive_language_sents.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`.. 
        * Put all the language negative class sentences in a text file named `negative_language_sents.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`..
        * Put all the library positive class sentences in a text file named `positive_library_sents.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`..
        * Put all the library negative class sentences in a text file named `negative_library_sents.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`..
        * Run the `training_pipeline/sentence_classifier_trainer.py` script.
    
    * For training Clustering Model
        * Put all the hardware positive class sentence embeddings in a text file named `positive_hardware_sent_embeddings.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`..
        * Put all the language positive class sentence embeddings in a text file named `positive_language_sent_embeddings.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`.. 
        * Put all the library positive class sentence embeddings in a text file named `positive_library_sent_embeddings.txt` inside `TRAINING_DATA_BASE_PATH`, a sample file has been included in the `sample_files/training_data`..
        * Run the `training_pipeline/clustering_trainer.py` 
    
    * For training NER Model:
        * Data present in `BILUO` format should be placed in `ner_trainer/data/cset`
        * Download Scibert Model and point the BERT_VOCAB and BERT_WEIGHTS variable in `ner_trainer/scripts/train_allennlp_local.sh` to the Scibert folder present inside `training_pipeline/ner_trainer/scibert`.
        * Point `dataset_size` variable in `ner_trainer/scripts/train_allennlp_local.sh` to length of training data.
        * From `ner_trainer` location, run `bash ./scripts/train_allennlp_local.sh output_dir`
        * output_dir is the directory where trained model files would be stored (can create an empty directory and point to it).  