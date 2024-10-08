"""
XNLI: Evaluating Cross-lingual Sentence Representations
https://arxiv.org/abs/1809.05053

Based on the implementation of @yongzx (see https://github.com/EleutherAI/lm-evaluation-harness/pull/258)

Prompt format (same as XGLM and mGPT):

sentence1 + ", right? " + mask = (Yes|Also|No) + ", " + sentence2

Predicition is the full sequence with the highest likelihood.

Language specific prompts are translated word-by-word with Google Translate
and may differ from the ones used by mGPT and XGLM (they do not provide their prompts).

"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean

_CITATIONS = """
@InProceedings{conneau2018xnli,
  author = "Conneau, Alexis
        and Rinott, Ruty
        and Lample, Guillaume
        and Williams, Adina
        and Bowman, Samuel R.
        and Schwenk, Holger
        and Stoyanov, Veselin",
  title = "XNLI: Evaluating Cross-lingual Sentence Representations",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  location = "Brussels, Belgium",
}
"""


class XNLIBase(Task):
    VERSION = 0
    DATASET_PATH = "xnli"
    DATASET_NAME = None

    QUESTION_WORD = None  # 'right'
    ENTAILMENT_LABEL = None  # 'Yes'
    NEUTRAL_LABEL = None  # 'Also'
    CONTRADICTION_LABEL = None  # 'No'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        # Example:
        # The girl that can help me is all the way across town, right? Yes, The girl I need help from lives a ways away.
        # [MASK] is replaced with ENTAILMENT_LABEL, NEUTRAL_LABEL, or CONTRADICTION_LABEL
        return (
            doc["premise"]
            + ", "
            + self.QUESTION_WORD
            + "? [MASK], "
            + doc["hypothesis"]
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return (
            " "
            + [self.ENTAILMENT_LABEL, self.NEUTRAL_LABEL, self.CONTRADICTION_LABEL][
                doc["label"]
            ]
        )

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        ll_true = rf.loglikelihood_rolling(ctx.replace("[MASK]", self.ENTAILMENT_LABEL))
        ll_neither = rf.loglikelihood_rolling(ctx.replace("[MASK]", self.NEUTRAL_LABEL))
        ll_false = rf.loglikelihood_rolling(
            ctx.replace("[MASK]", self.CONTRADICTION_LABEL)
        )

        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}


class XNLI_en(XNLIBase):  # English
    DATASET_NAME = "en"

    QUESTION_WORD = "right"
    ENTAILMENT_LABEL = "Yes"
    NEUTRAL_LABEL = "Also"
    CONTRADICTION_LABEL = "No"


class XNLI_de(XNLIBase):  # German
    DATASET_NAME = "de"

    QUESTION_WORD = "richtig"
    ENTAILMENT_LABEL = "Ja"
    NEUTRAL_LABEL = "Auch"
    CONTRADICTION_LABEL = "Nein"


class XNLI_ar(XNLIBase):  # Arabic
    DATASET_NAME = "ar"

    QUESTION_WORD = "صحيح"
    ENTAILMENT_LABEL = "نعم"
    NEUTRAL_LABEL = "لذا"
    CONTRADICTION_LABEL = "رقم"


class XNLI_bg(XNLIBase):  # Bulgarian
    DATASET_NAME = "bg"

    QUESTION_WORD = "правилно"
    ENTAILMENT_LABEL = "да"
    NEUTRAL_LABEL = "така"
    CONTRADICTION_LABEL = "не"


class XNLI_el(XNLIBase):  # Greek
    DATASET_NAME = "el"

    QUESTION_WORD = "σωστός"
    ENTAILMENT_LABEL = "Ναί"
    NEUTRAL_LABEL = "Έτσι"
    CONTRADICTION_LABEL = "όχι"


class XNLI_es(XNLIBase):  # Spanish
    DATASET_NAME = "es"

    QUESTION_WORD = "correcto"
    ENTAILMENT_LABEL = "Sí"
    NEUTRAL_LABEL = "Asi que"
    CONTRADICTION_LABEL = "No"


class XNLI_fr(XNLIBase):  # French
    DATASET_NAME = "fr"

    QUESTION_WORD = "correct"
    ENTAILMENT_LABEL = "Oui"
    NEUTRAL_LABEL = "Aussi"
    CONTRADICTION_LABEL = "Non"


# class XNLI_hi(XNLIBase):  # Hindi
#     DATASET_NAME = "hi"

#     QUESTION_WORD = 'right'
#     ENTAILMENT_LABEL = 'Yes'
#     NEUTRAL_LABEL = 'Also'
#     CONTRADICTION_LABEL = 'No'

#     QUESTION = "प्रश्न:"
#     ANSWER = "उत्तर:"
#     TRUE = "सत्य"
#     FALSE = "असत्य"
#     NEITHER = "तटस्थ"
#     OPTIONS = "सत्य या असत्य या तटस्थ?"


# class XNLI_ru(XNLIBase):  # Russian
#     DATASET_NAME = "ru"

#     QUESTION_WORD = 'right'
#     ENTAILMENT_LABEL = 'Yes'
#     NEUTRAL_LABEL = 'Also'
#     CONTRADICTION_LABEL = 'No'

#     QUESTION = "Вопрос:"
#     ANSWER = "Ответ:"
#     TRUE = "Правда"
#     FALSE = "Ложный"
#     NEITHER = "Нейтральный"
#     OPTIONS = "Правда, Ложный или Нейтральный?"


# class XNLI_sw(XNLIBase):  # Swahili
#     DATASET_NAME = "sw"

#     QUESTION_WORD = 'right'
#     ENTAILMENT_LABEL = 'Yes'
#     NEUTRAL_LABEL = 'Also'
#     CONTRADICTION_LABEL = 'No'

#     QUESTION = "Swali:"
#     ANSWER = "Jibu:"
#     TRUE = "Kweli"
#     FALSE = "Uongo"
#     NEITHER = "Wala"
#     OPTIONS = "Kweli, Uongo au Wala?"


# class XNLI_th(XNLIBase):  # Thai
#     DATASET_NAME = "th"

#     QUESTION_WORD = 'right'
#     ENTAILMENT_LABEL = 'Yes'
#     NEUTRAL_LABEL = 'Also'
#     CONTRADICTION_LABEL = 'No'

#     QUESTION = "คำถาม:"
#     ANSWER = "คำตอบ:"
#     TRUE = "จริง"
#     FALSE = "เท็จ"
#     NEITHER = "เป็นกลาง"
#     OPTIONS = "จริงหรือเท็จหรือเป็นกลาง?"


# class XNLI_tr(XNLIBase):  # Turkish
#     DATASET_NAME = "tr"

#     QUESTION_WORD = 'right'
#     ENTAILMENT_LABEL = 'Yes'
#     NEUTRAL_LABEL = 'Also'
#     CONTRADICTION_LABEL = 'No'

#     QUESTION = "Soru:"
#     ANSWER = "Cevap:"
#     TRUE = "Doğru"
#     FALSE = "Yanlış"
#     NEITHER = "Nötr"
#     OPTIONS = "Doğru, Yanlış veya Nötr?"


# class XNLI_ur(XNLIBase):  # Urdu
#     DATASET_NAME = "ur"

#     QUESTION_WORD = 'right'
#     ENTAILMENT_LABEL = 'Yes'
#     NEUTRAL_LABEL = 'Also'
#     CONTRADICTION_LABEL = 'No'

#     QUESTION = ":سوال"
#     ANSWER = ":جواب"
#     TRUE = "صحیح"
#     FALSE = "غلط"
#     NEITHER = "غیر جانبدار"
#     OPTIONS = "صحیح یا غلط یا غیر جانبدار؟"


# class XNLI_vi(XNLIBase):  # Vietnamese
#     DATASET_NAME = "vi"

#     QUESTION = "Câu hỏi:"
#     ANSWER = "Câu trả lời:"
#     TRUE = "Đúng"
#     FALSE = "Sai"
#     NEITHER = "Trung lập"
#     OPTIONS = "Đúng, Sai hay Trung lập?"


# class XNLI_zh(XNLIBase):  # Chinese
#     DATASET_NAME = "zh"

#     QUESTION_WORD = 'right'
#     ENTAILMENT_LABEL = 'Yes'
#     NEUTRAL_LABEL = 'Also'
#     CONTRADICTION_LABEL = 'No'

#     QUESTION = "问题:"
#     ANSWER = "回答:"
#     TRUE = "对"
#     FALSE = "错"
#     NEITHER = "中立"
#     OPTIONS = "对、错、还是中立?"

LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    # "hi",
    # "ru",
    # "sw",
    # "th",
    # "tr",
    # "ur",
    # "vi",
    # "zh",
]

LANG_CLASSES = [
    XNLI_ar,
    XNLI_bg,
    XNLI_de,
    XNLI_el,
    XNLI_en,
    XNLI_es,
    XNLI_fr,
    # XNLI_hi,
    # XNLI_ru,
    # XNLI_sw,
    # XNLI_th,
    # XNLI_tr,
    # XNLI_ur,
    # XNLI_vi,
    # XNLI_zh,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"xnli_{lang}"] = lang_class
    return tasks
