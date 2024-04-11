# TODO consider if this can be collapsed back down into the pipeline_train.py
import argparse
import json
import logging
import random
import os

from sklearn.metrics import accuracy_score

from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer
import sys
sys.path.insert(1, 'D:\Bert_ex\Transformer-Explainability')
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator

from BERT_rationale_benchmark.utils import (
    Annotation,
    Evidence,
    write_jsonl,
    load_datasets,
    load_documents,
)
from BERT_explainability.modules.BERT.BertForSequenceClassification import \
    BertForSequenceClassification as BertForSequenceClassificationTest
from BERT_explainability.modules.BERT.BERT_cls_lrp import \
    BertForSequenceClassification as BertForClsOrigLrp

from transformers import BertForSequenceClassification

from collections import OrderedDict

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)
# let's make this more or less deterministic (not resistent to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
    print("begin generate -------------------------")
    attention_list = attention_list[:len(text_list)]
    if attention_list.max() == attention_list.min():
        attention_list = torch.zeros_like(attention_list)
    else:
        attention_list = 100 * (attention_list - attention_list.min()) / (attention_list.max() - attention_list.min())
    attention_list[attention_list < 1] = 0
    attention_list = attention_list.tolist()
    text_list = [text_list[i].replace('$', '') for i in range(len(text_list))]
    if rescale_value:
        print("----begin rescale--------")
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth=150mm]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            # string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
            # print(text_list[idx])
            if '\#\#' in text_list[idx]:
                token = text_list[idx].replace('\#\#', '')
                string += "\\colorbox{%s!%s}{" % (color, attention_list[idx]) + "\\strut " + token + "}"
            else:
                string += " " + "\\colorbox{%s!%s}{" % (color, attention_list[idx]) + "\\strut " + text_list[idx] + "}"
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}
\end{document}''')

def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list


def scores_per_word_from_scores_per_token(input, tokenizer, input_ids, scores_per_id):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    words = [word.replace('##', '') for word in words]
    score_per_char = []

    # TODO: DELETE
    input_ids_chars = []
    for word in words:
        if word in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        input_ids_chars += list(word)
    # TODO: DELETE

    for i in range(len(scores_per_id)):
        if words[i] in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        score_per_char += [scores_per_id[i]] * len(words[i])


    score_per_word = []
    start_idx = 0
    end_idx = 0
    # TODO: DELETE
    words_from_chars = []
    for inp in input:
        if start_idx >= len(score_per_char):
            break
        end_idx = end_idx + len(inp)
        score_per_word.append(np.max([score.detach().cpu().numpy() for score in score_per_char[start_idx:end_idx]]))

        # TODO: DELETE
        words_from_chars.append(''.join(input_ids_chars[start_idx:end_idx]))

        start_idx = end_idx

    if (words_from_chars[:-1] != input[:len(words_from_chars)-1]):
        print(words_from_chars)
        print(input[:len(words_from_chars)])
        print(words)
        print(tokenizer.convert_ids_to_tokens(input_ids))
        assert False

    return torch.tensor(score_per_word)

def get_input_words(input, tokenizer, input_ids):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    words = [word.replace('##', '') for word in words]

    input_ids_chars = []
    for word in words:
        if word in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        input_ids_chars += list(word)

    start_idx = 0
    end_idx = 0
    words_from_chars = []
    for inp in input:
        if start_idx >= len(input_ids_chars):
            break
        end_idx = end_idx + len(inp)
        words_from_chars.append(''.join(input_ids_chars[start_idx:end_idx]))
        start_idx = end_idx

    if (words_from_chars[:-1] != input[:len(words_from_chars)-1]):
        print(words_from_chars)
        print(input[:len(words_from_chars)])
        print(words)
        print(tokenizer.convert_ids_to_tokens(input_ids))
        assert False
    return words_from_chars

def bert_tokenize_doc(doc: List[List[str]], tokenizer, special_token_map) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
    """ Tokenizes a document and returns [start, end) spans to map the wordpieces back to their source words"""
    sents = []
    sent_token_spans = []
    for sent in doc:
        tokens = []
        spans = []
        start = 0
        for w in sent:
            if w in special_token_map:
                tokens.append(w)
            else:
                tokens.extend(tokenizer.tokenize(w))
            end = len(tokens)
            spans.append((start, end))
            start = end
        sents.append(tokens)
        sent_token_spans.append(spans)
    return sents, sent_token_spans

def initialize_models(params: dict, batch_first: bool, use_half_precision=False):
    assert batch_first
    max_length = params['max_length']
    tokenizer = BertTokenizer.from_pretrained(params['bert_vocab'])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    bert_dir = params['bert_dir']
    evidence_classes = dict((y, x) for (x, y) in enumerate(params['evidence_classifier']['classes']))
    evidence_classifier = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=len(evidence_classes))
    word_interner = tokenizer.vocab
    de_interner = tokenizer.ids_to_tokens
    return evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer


BATCH_FIRST = True


def extract_docid_from_dataset_element(element):
    return next(iter(element.evidences))[0].docid

def extract_evidence_from_dataset_element(element):
    return next(iter(element.evidences))


def main():
    parser = argparse.ArgumentParser(description="""Trains a pipeline model.

    Step 1 is evidence identification, that is identify if a given sentence is evidence or not
    Step 2 is evidence classification, that is given an evidence sentence, classify the final outcome for the final task
     (e.g. sentiment or significance).

    These models should be separated into two separate steps, but at the moment:
    * prep data (load, intern documents, load json)
    * convert data for evidence identification - in the case of training data we take all the positives and sample some
      negatives
        * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a
          broader sampling of negative values.
    * train evidence identification
    * convert data for evidence classification - take all rationales + decisions and use this as input
    * train evidence classification
    * decode first the evidence, then run classification for each split
    
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_params', dest='model_params', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    args = parser.parse_args()
    assert BATCH_FIRST
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.model_params, 'r') as fp:
        logger.info(f'Loading model parameters from {args.model_params}')
        model_params = json.load(fp)
        logger.info(f'Params: {json.dumps(model_params, indent=2, sort_keys=True)}')
    train, val, test = load_datasets(args.data_dir)
    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(test)))))
    documents = load_documents(args.data_dir, docids)
    logger.info(f'Loaded {len(documents)} documents')
    evidence_classes = dict((y, x) for (x, y) in enumerate(model_params['evidence_classifier']['classes']))
    tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")
    word_interner = tokenizer.vocab
    logger.info(f'We have {len(word_interner)} wordpieces')
    
    cache = os.path.join(args.output_dir, 'preprocessed1.pkl')
    if os.path.exists(cache):
        logger.info(f'Loading interned documents from {cache}')
        (interned_documents) = torch.load(cache)
    else:
        logger.info(f'Interning documents')
        interned_documents = {}
        for d, doc in documents.items():
            print("------------doc--------")
            print(d, doc)
            encoding = tokenizer(doc, return_tensors='pt')
            interned_documents[d] = encoding
        torch.save((interned_documents), cache)

    save_dir = args.output_dir

    evidence_classifier_output_dir = os.path.join(save_dir, 'classifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_classifier_output_dir, exist_ok=True)
    model_save_file = os.path.join(evidence_classifier_output_dir, 'SecBert_6labels_balance.pt')
    epoch_save_file = os.path.join(evidence_classifier_output_dir, 'classifier_epoch_data.pt')
    
    
    
    # test

    test_classifier = BertForSequenceClassificationTest.from_pretrained(model_params['bert_dir'],
                                                                        num_labels=6).to('cuda')
    orig_lrp_classifier = BertForClsOrigLrp.from_pretrained(model_params['bert_dir'],
                                                            num_labels=6).to('cuda')
    
    # fix model save file
    checkpoint = torch.load(model_save_file)
    state_dict = checkpoint['state_dict']
    dict1 = state_dict.copy()
    for name in state_dict.keys():
        name_new = name.replace('bert_model', 'bert')
        dict1[name_new] = state_dict[name]
        del dict1[name]
    dict2 = test_classifier.state_dict()
    dict1["bert.embeddings.position_embeddings.weight"] = state_dict["bert_model.embeddings.position_embeddings.weight"]
    dict1["bert.embeddings.word_embeddings.weight"] = state_dict["bert_model.embeddings.word_embeddings.weight"]
    dict1["classifier.weight"] = state_dict["linear.weight"]
    dict1["classifier.bias"] = state_dict["linear.bias"]
    dict1["bert.embeddings.position_ids"] = dict2["bert.embeddings.position_ids"]
    #end
    
    
    classifications = ["000 - Normal", '126 - Path Traversal',
               '242 - Code Injection', '274 - HTTP Verb Tampering',
               '66 - SQL Injection', '88 - OS Command Injection']
    if os.path.exists(epoch_save_file):
        logging.info(f'Restoring model from {model_save_file}')
        

        
        test_classifier.load_state_dict(dict1)
        orig_lrp_classifier.load_state_dict(dict1)
        test_classifier.eval()
        orig_lrp_classifier.eval()
        test_batch_size = 1
        logging.info(
            f'Testing with {len(test) // test_batch_size} batches with {len(test)} examples')

        # explainability
        explanations = Generator(test_classifier)
        explanations_orig_lrp = Generator(orig_lrp_classifier)
        method = "ground_truth"
        method_folder = {"transformer_attribution": "ours", "partial_lrp": "partial_lrp", "last_attn": "last_attn",
                         "attn_gradcam": "attn_gradcam", "lrp": "lrp", "rollout": "rollout",
                         "ground_truth": "ground_truth", "generate_all": "generate_all"}
        method_expl = {"transformer_attribution": explanations.generate_LRP,
                       "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
                       "last_attn": explanations_orig_lrp.generate_attn_last_layer,
                       "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
                       "lrp": explanations_orig_lrp.generate_full_lrp,
                       "rollout": explanations_orig_lrp.generate_rollout}

        os.makedirs(os.path.join(args.output_dir, method_folder[method]), exist_ok=True)

        result_files = []
        for i in range(1,11,1):
            result_files.append(open(os.path.join(args.output_dir, '{0}/identifier_results_{1}.json').format(method_folder[method], i), 'w'))

        j = 0
        for batch_start in range(0, len(test), test_batch_size):
            batch_elements = test[batch_start:min(batch_start + test_batch_size, len(test))]
            print("---batch_elment----")
            print(batch_elements)
            targets = [evidence_classes[s.classification] for s in batch_elements]
            targets = torch.tensor(targets, dtype=torch.long, device='cuda')
            samples_encoding = [interned_documents[extract_docid_from_dataset_element(s)] for s in batch_elements]
            print(samples_encoding)
            input_ids = torch.stack(
                [samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(1).to('cuda')
            attention_masks = torch.stack(
                [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(
                'cuda')
            preds = test_classifier(input_ids=input_ids, attention_mask=attention_masks)[0]

            for s in batch_elements:
                doc_name = extract_docid_from_dataset_element(s)
                inp = documents[doc_name].lower()
                classification = classifications[targets.item()]
                is_classification_correct = 1 if preds.argmax(dim=1) == targets else 0
                print(is_classification_correct)
                if method == "ground_truth":
                    inp_cropped = tokenizer.convert_ids_to_tokens(input_ids[0])
                    print(inp_cropped)
                    cam = torch.zeros(len(inp_cropped))
                    for evidence in extract_evidence_from_dataset_element(s):
                        start_idx = evidence.start_token
                        if start_idx >= len(cam):
                            break
                        end_idx = evidence.end_token
                        cam[start_idx:end_idx] = 1
                    generate(inp_cropped, cam,
                             (os.path.join(args.output_dir, '{0}/visual_results_{1}.tex').format(method_folder[method],
                                                                                                 j)), color="green")
                    j = j + 1
                    break
                text = tokenizer.convert_ids_to_tokens(input_ids[0])
                print(text)
                target_idx = targets.item()
                cam_target = method_expl[method](input_ids=input_ids, attention_mask=attention_masks, index=target_idx, start_layer=0)[0]
                cam_target = cam_target.clamp(min=0)
                generate(text, cam_target,
                         (os.path.join(args.output_dir, '{0}/{1}_GT_{2}_{3}.tex').format(
                             method_folder[method], j, classification, is_classification_correct)))
                if method in ["transformer_attribution", "partial_lrp", "attn_gradcam", "lrp"]:
                    cam_false_class = method_expl[method](input_ids=input_ids, attention_mask=attention_masks, index=1-target_idx, start_layer=0)[0]
                    cam_false_class = cam_false_class.clamp(min=0)
                    generate(text, cam_false_class,
                         (os.path.join(args.output_dir, '{0}/{1}_CF.tex').format(
                             method_folder[method], j)))
                cam = cam_target
                # cam = scores_per_word_from_scores_per_token(inp, tokenizer,input_ids[0], cam)
                print(cam)
                j = j + 1
                doc_name = extract_docid_from_dataset_element(s)
                hard_rationales = []
                for res, i in enumerate(range(1, 11, 1)):
                    print("calculating top ", i)
                    _, indices = cam.topk(k=i)
                    for index in indices.tolist():
                        hard_rationales.append({
                            "start_token": index,
                            "end_token": index+1
                        })
                    result_dict = {
                        "annotation_id": doc_name,
                        "rationales": [{
                            "docid": doc_name,
                            "hard_rationale_predictions": hard_rationales
                        }],
                    }
                    result_files[res].write(json.dumps(result_dict) + "\n")

        for i in range(len(result_files)):
            result_files[i].close()


if __name__ == '__main__':
    main()
