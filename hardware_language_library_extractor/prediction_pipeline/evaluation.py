import os

from hardware_language_library_extractor.common.util import load_data_from_json, read_txt_into_df, get_alias_name
from hardware_language_library_extractor.prediction_pipeline.config import OUTPUT_FOLDER, TRAINING_DATA_BASE_PATH, \
    ANNOTATED_FOLDER, ANNOTATED_PAPERS_LIST, OUTPUT_PAPER_LIST
from hardware_language_library_extractor.logger import Logger

logger = Logger("evaluation").logger


def match_sent_units(sent_unit1, sent_unit2, regex_pattern):
    if sent_unit1['sent'] == sent_unit2['sentence'] and regex_pattern.search(sent_unit2['entity']):
        return True
    return False


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall


def get_all_entities(paper_data):
    entities = set()
    for sent_unit in paper_data["hardware"]:
        if "entity" in sent_unit and sent_unit["entity_type"] == "hardware":
            entities.add(sent_unit["entity"])
        elif "entities" in sent_unit:
            entities.update(sent_unit["entities"])
    return entities


def match_pdf_outputs(paper_pipeline_name, annotated_paper_name):
    paper_pipeline = load_data_from_json(os.path.join(OUTPUT_FOLDER, paper_pipeline_name))
    paper_annotation_output = load_data_from_json(
        os.path.join(TRAINING_DATA_BASE_PATH, ANNOTATED_FOLDER, annotated_paper_name))
    tp, fp, fn = 0, 0, 0
    annotated_entities = get_all_entities(paper_annotation_output)
    pipeline_entities = get_all_entities(paper_pipeline)
    fn = len(annotated_entities - pipeline_entities)
    tp = len(annotated_entities.intersection(pipeline_entities))
    fp = len(pipeline_entities - annotated_entities)
    # for sent_unit_pa in paper_annotation_output["hardware"]:
    #     matched = False
    #     sent_unit_pa['sent'] = preprocessing_single_sentence(sent_unit_pa['sent'])
    #     regex_pattern = get_regex_pattern(sent_unit_pa["entities"])
    #     for sent_unit_pp in paper_pipeline["hardware"]:
    #         if sent_unit_pa['sent'] == sent_unit_pp['sentence']:
    #             if match_sent_units(sent_unit_pa, sent_unit_pp, regex_pattern):
    #                 tp += 1
    #             else:
    #                 fp += 1
    #             matched = True
    #     if not matched:
    #         fn += 1
    precision, recall = calculate_metrics(tp, fp, fn)
    logger.info("Individual Paper {} : Precision: {}, Recall: {}".format(annotated_paper_name, precision, recall))
    return tp, fp, fn


def init():
    global all_annotated_paper
    all_annotated_paper = sorted(
        read_txt_into_df(os.path.join(os.path.join(TRAINING_DATA_BASE_PATH, ANNOTATED_FOLDER, ANNOTATED_PAPERS_LIST)))[0])
    all_annotated_paper = [get_alias_name(name) for name in all_annotated_paper]


def driver(paper_pipeline_name):
    paper_processed = False
    tp, fp, fn = 0, 0, 0
    alias_name = get_alias_name(paper_pipeline_name)
    if alias_name in all_annotated_paper:
        paper_processed = True
        tp, fp, fn = match_pdf_outputs(paper_pipeline_name, "{}.json".format(alias_name))

    return paper_processed, (tp, fp, fn)


def main():
    init()
    ttp, tfp, tfn, processed_paper_cnt = 0, 0, 0, 0
    output_paper_list = read_txt_into_df(os.path.join(os.path.join(OUTPUT_FOLDER, OUTPUT_PAPER_LIST)))[0]
    for paper in output_paper_list:
        processed, (tp, fp, fn) = driver(paper)
        if processed:
            processed_paper_cnt += 1
        ttp += tp
        tfp += fp
        tfn += fn
    precision, recall = calculate_metrics(ttp, tfp, tfn)
    print("Overall: processed_paper_cnt: {}, Precision: {}, Recall: {}".format(processed_paper_cnt, precision, recall))
    logger.info(
        "Overall: processed_paper_cnt: {}, Precision: {}, Recall: {}".format(processed_paper_cnt, precision, recall))


if __name__ == '__main__':
    main()
