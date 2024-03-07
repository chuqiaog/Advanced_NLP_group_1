# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This metric calculates both Token Overlap and Span Agreement precision, recall and f1 scores."""

import datasets



_CITATION = """\
@inproceedings{morante-blanco-2012-sem,
title = "*{SEM} 2012 Shared Task: Resolving the Scope and Focus of Negation",
author = "Morante, Roser and Blanco, Eduardo",
booktitle = "*{SEM} 2012: The First Joint Conference on Lexical and Computational Semantics {--} Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation ({S}em{E}val 2012)",
month = "7-8 " # jun,
year = "2012",
address = "Montr{\'e}al, Canada",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/S12-1035",
pages = "265--274",
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This metric calculates both Token Overlap and Span Agreement precision, recall and f1 scores. This script is adapted from seqeval.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: List of List of predicted labels.
    references: List of List of reference labels.
Returns:
    'token_precision': precision,
    'token_recall': recall,
    'token_f1': F1 score for token overlap
    
    'span_precision': precision,
    'span_recall': recall,
    'span_f1': F1 score for span agreement

"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SpanAgree(datasets.Metric):
    """Calculates span agreement metric."""

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,

            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value("int8", id="label"), id="sequence"),
                "references": datasets.Sequence(datasets.Value("int8", id="label"), id="sequence"),
            }),

            homepage="https://github.com/dannashao/appliedTM_C",
            codebase_urls=["https://github.com/dannashao/appliedTM_C"],
            reference_urls=["https://github.com/dannashao/appliedTM_C"]
        )


    def _compute(self, predictions, references):
        """Returns the scores"""
        # TOKEN LEVEL
        tn, fp, fn, tp = 0,0,0,0
        for span_true, span_pred in zip(references, predictions):
            for token_true, token_pred in zip(span_true, span_pred):
                if token_true == 1:
                    if token_pred == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if token_pred == 1:
                        fp += 1
                    else:
                        tn += 1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # SPAN LEVEL
        tn, fp, fn, tp = 0,0,0,0
        for span_true, span_pred in zip(references, predictions):
            if 1 in span_true:
                if span_true == span_pred:
                    tp += 1
                elif all([(yt == 0 or (yt == 1 and predictions[i] == 1)) for i, yt in enumerate(references)]):
                    fp += 1
                else:
                    fp += 1
                    fn += 1
            else:
                if 1 in span_pred:
                    fp += 1
                    fn += 1
                else:
                    tn += 1
                    
        span_precision = tp / (tp + fp) if tp + fp > 0 else 0
        span_recall = tp / (tp + fn) if tp + fn > 0 else 0
        span_f1 = 2 * (span_precision * span_recall) / (span_precision + span_recall) if span_precision + span_recall > 0 else 0

        scores = {"token_precision":precision, "token_recall":recall, "token_f1":f1,
                  "span_precision":span_precision, "span_recall":span_recall, "span_f1":span_f1}
        return scores