import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM

from asl_utils import combine_sequences
from sklearn.model_selection import KFold


class ModelSelector(object):
    """
    base class for model selection (strategy design pattern)
    """

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        min_bic_score = np.Infinity
        best_model = None
        for n_state in range(self.min_n_components, self.max_n_components):
            word_sequences = self.words
            model = GaussianHMM(n_components=n_state, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False)
            try:
                model.fit(self.X, self.lengths)
                log_likelihood = model.score(self.X)
            except:
                continue
            num_parameters = 2 * len(model.means_[0]) * n_state + n_state * n_state - 1
            bic_score = -2 * log_likelihood + num_parameters * np.log(len(word_sequences))
            if bic_score < min_bic_score:
                min_bic_score = bic_score
                best_model = model
        return best_model


class SelectorDIC(ModelSelector):
    """ select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        min_dic_score = np.Infinity
        best_model = None
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                log_likelihood = model.score(self.X, self.lengths)
                total_other_log_likelihood = 0
                for word in self.words:
                    other_x, other_lengths = self.hwords[word]
                    total_other_log_likelihood += model.score(other_x, other_lengths)
                avg_log_likelihood = total_other_log_likelihood / (len(self.words) - 1)
                dic_score = log_likelihood - avg_log_likelihood
                if dic_score < min_dic_score:
                    min_dic_score = dic_score
                    best_model = model
            except:
                continue
        return best_model


class SelectorCV(ModelSelector):
    """ select best model based on average log Likelihood of cross-validation folds

    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = None
        max_avg_log_likelihood = -np.log(np.Infinity)
        for n_state in range(self.min_n_components, self.max_n_components + 1):
            n_splits = 3
            if len(self.words[self.this_word]) < n_splits:
                n_splits = 2
            split_method = KFold(n_splits=n_splits)
            total_log_likelihood = 0
            model = GaussianHMM(n_components=n_state, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False)
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.words[self.this_word]):
                    X, lengths = combine_sequences(cv_train_idx, sequences=self.words[self.this_word])
                    model.fit(X, lengths=lengths)
                    y, _ = combine_sequences(cv_test_idx, sequences=self.words[self.this_word])
                    log_likelihood = model.score(y)
                    total_log_likelihood += log_likelihood
            except ValueError:
                continue
            avg_log_likelihood = total_log_likelihood / n_splits
            if max_avg_log_likelihood < avg_log_likelihood:
                max_avg_log_likelihood = avg_log_likelihood
                best_model = model
        return best_model
