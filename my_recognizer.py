import warnings

import numpy

from asl_data import SinglesData
from collections import OrderedDict


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    all_Xlengths = test_set.get_all_Xlengths()
    for index in all_Xlengths:
        prob_dict = {}
        y = all_Xlengths[index][0]
        lengths = all_Xlengths[index][1]
        for model in models:
            try:
                prob_dict[model] = models[model].score(y, lengths)
            except:
                prob_dict[model] = -numpy.Infinity
        probabilities.append(prob_dict)
    for prob_dict in probabilities:
        max_log_likelihood = -numpy.log(numpy.Infinity)
        best_guess = None
        for word in prob_dict:
            if float(prob_dict[word]) > max_log_likelihood:
                max_log_likelihood = float(prob_dict[word])
                best_guess = word
        guesses.append(best_guess)
    return probabilities, guesses
