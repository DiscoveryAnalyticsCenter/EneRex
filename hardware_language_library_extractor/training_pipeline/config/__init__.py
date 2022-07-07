import yaml
import os

if 'HLL_EXTRACTOR_SETTINGS' in os.environ:
    environ = os.environ["HLL_EXTRACTOR_SETTINGS"]
else:
    raise Exception('Environment variable not set, kindly set HLL_EXTRACTOR_SETTINGS to either local or remote')
if environ == 'local':
    stream = open('/mnt/c/Users/kkkoo/OneDrive/Documents/VirginiaTech/DACResearch/Scripts/pipeline/hardware_language_library_extractor/prediction_pipeline/config/local.yml', 'r')
elif environ == 'remote':
    stream = open(os.path.normpath('hardware_language_library_extractor/prediction_pipeline/config/remote.yml'), 'r')
else:
    raise Exception('Incorrect value of HLL_EXTRACTOR_SETTINGS, accepted value is either local or remote')

config = yaml.load(stream, Loader=yaml.SafeLoader)

HARDWARE_SENTENCE_CLASSIFIER = "transformer_h_classifier"
LANGUAGE_SENTENCE_CLASSIFIER = "transformer_lang_classifier"
LIBRARY_SENTENCE_CLASSIFIER = "transformer_lib_classifier"
HARDWARE_CLUSTERING_MODEL = "hardware_clustering_model.sav"
LANGUAGE_CLUSTERING_MODEL = "language_clustering_model.sav"
LIBRARY_CLUSTERING_MODEL = "library_clustering_model.sav"
HARDWARE_POSITIVE_SENTENCES = "positive_hardware_sents.txt"
HARDWARE_NEGATIVE_SENTENCES = "negative_hardware_sents.txt"
LANGUAGE_POSITIVE_SENTENCES = "positive_language_sents.txt"
LANGUAGE_NEGATIVE_SENTENCES = "negative_language_sents.txt"
LIBRARY_POSITIVE_SENTENCES = "positive_library_sents.txt"
LIBRARY_NEGATIVE_SENTENCES = "negative_library_sents.txt"
HARDWARE_SENTENCE_EMBEDDINGS = "positive_hardware_sent_embeddings.txt"
LANGUAGE_SENTENCE_EMBEDDINGS = "positive_language_sent_embeddings.txt"
LIBRARY_SENTENCE_EMBEDDINGS = "positive_library_sent_embeddings.txt"
NER_MODEL = "bert_crf_ner"
SCIBERT_EMBEDDING_MODEL = "scibert_scivocab_uncased"
LOGGER_NAME = "extraction_pipeline"
LOG_LEVEL = "INFO"
MIN_THRESHOLD_SENTCHAR_LEN = 20
MAX_THRESHOLD_SENTCHAR_LEN = 250
SENTENCE_CLASSIFIER_PROBABILITY_THRESHOLD = 0.95
CLUSTER_IMAGES = "cluster_images"

PDF_FOLDER_PATH = config['PDF_FOLDER_PATH']
LIST_OF_PDFs = config['LIST_OF_PDFs']
OUTPUT_FOLDER_BASE_PATH = config['OUTPUT_FOLDER']
MODELS_FOLDER_BASE_PATH = config['MODELS_FOLDER_BASE_PATH']
TRAINING_DATA_BASE_PATH = config['TRAINING_DATA_BASE_PATH']

# Transformer based Sentence Classifier training parameters
OVERWRITE_OUTPUT_DIRECTORY = True
DO_TRAIN = True
DO_EVAL = True
PER_DEVICE_TRAIN_BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH_SIZE = 128
NUM_TRAIN_EPOCHS = 10
LOGGING_STEPS = 500
LOGGING_FIRST_STEP = True
SAVE_STEPS = 1000
EVALUATE_DURING_TRAINING = True
