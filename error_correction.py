from smoothing_classes import *
from config import error_correction
import numpy as np
import pandas as pd

class SpellingCorrector:

    def __init__(self):

        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']

        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK()
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])

    def fit(self, data: List[str]) -> None:
        """
        Fit the spelling corrector model to the data.
        :param data: The input data.
        """
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)

    def correct(self, text: List[str]) -> List[str]:
        """
        Correct the input text.
        :param text: The input text.
        :return: The corrected text.
        """
        ## there will be an assertion to check if the output text is of the same
        ## length as the input text
        return text
