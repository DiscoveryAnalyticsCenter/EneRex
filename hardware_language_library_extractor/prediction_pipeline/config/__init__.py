import yaml
import os

if 'HLL_EXTRACTOR_SETTINGS' in os.environ:
    environ = os.environ["HLL_EXTRACTOR_SETTINGS"]
else:
    raise Exception('Environment variable not set, kindly set HLL_EXTRACTOR_SETTINGS to either local or remote')
if environ == 'local':
    stream = open(os.path.normpath('hardware_language_library_extractor/prediction_pipeline/config/local.yml'), 'r')
elif environ == 'remote':
    stream = open(os.path.normpath('hardware_language_library_extractor/prediction_pipeline/config/remote.yml'), 'r')
else:
    raise Exception('Incorrect value of HLL_EXTRACTOR_SETTINGS, accepted value is either local or remote')

config = yaml.load(stream, Loader=yaml.SafeLoader)

HARDWARE_CLASSIFIER_MODEL = "hardware_sentence_classifier.sav"
TRANSFORMER_HARDWARE_CLASSIFIER = "transformer_h_classifier/checkpoint-12000"
TRANSFORMER_LANGUAGE_CLASSIFIER = "transformer_lang_classifier/checkpoint-14000"
TRANSFORMER_LIBRARY_CLASSIFIER = "transformer_lib_classifier/checkpoint-12000"
LANGUAGE_CLASSIFIER_MODEL = "language_sentence_classifier.sav"
LIBRARY_CLASSIFIER_MODEL = "library_sentence_classifier.sav"
HARDWARE_CLUSTERING_MODEL = "hardware_clustering_model.sav"
LANGUAGE_CLUSTERING_MODEL = "language_clustering_model.sav"
LIBRARY_CLUSTERING_MODEL = "library_clustering_model.sav"
NER_MODEL = "bert_crf_ner"
SCIBERT_EMBEDDING_MODEL = "scibert_scivocab_uncased"
LOGGER_NAME = "extraction_pipeline"
LOG_LEVEL = "INFO"
MIN_THRESHOLD_SENTCHAR_LEN = 20
MAX_THRESHOLD_SENTCHAR_LEN = 250
SENTENCE_CLASSIFIER_PROBABILITY_THRESHOLD = 0.95

# Used for evaluating the prediction
ANNOTATED_FOLDER = "annotated"
ANNOTATED_PAPERS_LIST = "annotated_paper.txt"
OUTPUT_PAPER_LIST = "output_list.txt"

PDF_FOLDER_PATH = config['PDF_FOLDER_PATH']
LIST_OF_PDFs = config['LIST_OF_PDFs']
OUTPUT_FOLDER = config['OUTPUT_FOLDER']
MODELS_FOLDER_BASE_PATH = config['MODELS_FOLDER_BASE_PATH']
TRAINING_DATA_BASE_PATH = config['TRAINING_DATA_BASE_PATH']
