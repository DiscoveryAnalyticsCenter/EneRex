import yaml
import os

if 'HLL_EXTRACTOR_SETTINGS' in os.environ:
    environ = os.environ["HLL_EXTRACTOR_SETTINGS"]
else:
    raise Exception('Environment variable not set, kindly set HLL_EXTRACTOR_SETTINGS to either local or remote')
if environ == 'local':
    stream = open(
        '/mnt/c/Users/kkkoo/OneDrive/Documents/VirginiaTech/DACResearch/Scripts/pipeline/hardware_language_library_extractor/prediction_pipeline/config/local.yml',
        'r')
elif environ == 'remote':
    stream = open(os.path.normpath('hardware_language_library_extractor/prediction_pipeline/config/remote.yml'), 'r')
else:
    raise Exception('Incorrect value of HLL_EXTRACTOR_SETTINGS, accepted value is either local or remote')

config = yaml.load(stream, Loader=yaml.SafeLoader)

HARDWARE_CLUSTER_MAPPING_FILE = 'spacyhardware_sentence_mapping.csv'
LANGUAGE_CLUSTER_MAPPING_FILE = 'spacylanguage_sentence_mapping.csv'
LIBRARY_CLUSTER_MAPPING_FILE = 'spacylibrary_sentence_mapping.csv'
HARDWARE_POSITIVE_SENTENCES = "positive_hardware_sents.txt"
HARDWARE_NEGATIVE_SENTENCES = "negative_hardware_sents.txt"
LANGUAGE_POSITIVE_SENTENCES = "positive_language_sents.txt"
LANGUAGE_NEGATIVE_SENTENCES = "negative_language_sents.txt"
LIBRARY_POSITIVE_SENTENCES = "positive_library_sents.txt"
LIBRARY_NEGATIVE_SENTENCES = "negative_library_sents.txt"
SPACY_NER_TRAIN_DATA = "spacy_ner.jsonl"
ANNOTATED_FOLDER = "annotated"
ANNOTATED_PAPERS_LIST = "annotated_paper.txt"
OUTPUT_PAPER_LIST = "output_list.txt"
LOGGER_NAME = "extraction_pipeline"
LOG_LEVEL = "INFO"
MIN_THRESHOLD_SENTCHAR_LEN = 20
MAX_THRESHOLD_SENTCHAR_LEN = 250
SENTENCE_CLASSIFIER_PROBABILITY_THRESHOLD = 0.95
HARDWARE_KEYWORDS = ["CPU", "Kernels", "NVIDIA", "FPGA", "Core", "GPU", "Tensor Core", "Tensor Processing Unit", "CUDA",
                     "GHz", "AMD", "Dell", "Precision", "registers", "SIMD", "processor",
                     "Linux", "RAM", "Intel", "Xeon"]
LANGUAGE_KEYWORDS = ['python', 'matlab', 'Java', 'R']
LIBRARY_KEYWORDS = ["scikit", "pytorch", "opennmt", "tensorflow", "keras", "theano", "caffe", "torch", "mxnet",
                    "coreML", "CNTK"]

PDF_FOLDER_PATH = config['PDF_FOLDER_PATH']
LIST_OF_PDFs = config['LIST_OF_PDFs']
OUTPUT_FOLDER_BASE_PATH = config['OUTPUT_FOLDER']
MODELS_FOLDER_BASE_PATH = config['MODELS_FOLDER_BASE_PATH']
TRAINING_DATA_BASE_PATH = config['TRAINING_DATA_BASE_PATH']
