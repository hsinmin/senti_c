# -*- coding: utf-8 -*-
from .sentence_processor import SentenceProcessor,sentence_convert_examples_to_features,sigmoid_array

from .aspect_processor import AspectProcessor,aspect_convert_examples_to_features,tag2ot,tag2ts,evaluate_ote,evaluate_ts,ASPECT_LABEL_MAP,SENTIMENT_LABEL_MAP,chg_labels_to_aspect_and_sentiment

from .general import split_text_from_input_data,get_toolkit_models,get_domain_embedding