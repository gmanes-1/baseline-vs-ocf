"""Single implementation of the symbolic MONK DNF Baseline learner."""

from itertools import combinations

import numpy as np


class BaselineFormulaLearner:
    """Learn a disjunction of safe conjunctive attribute-value terms.

    Repeated feature vectors are aggregated using delta = positive count -
    negative count. A vector is positive when delta >= 0 and negative
    otherwise. Candidate terms are generated from nonempty subsets of
    positive-vector attributes, filtered to exclude every negative vector,
    and selected greedily by uncovered-positive coverage.

    Ties prefer fewer literals, then lexicographically smaller attribute
    indices and stringified categorical values.
    """

    def __init__(self):
        self.n_features = None
        self.positive_rows = set()
        self.negative_rows = set()
        self.terms = set()
        self.uncovered_positives = set()
        self.conflicting_patterns = set()

    @staticmethod
    def _term_matches(term, row):
        cols, values = term
        return all(row[col] == value for col, value in zip(cols, values))

    def _rebuild_formula(self):
        self.conflicting_patterns = self.positive_rows & self.negative_rows
        positives = sorted(self.positive_rows - self.negative_rows)
        negatives = self.negative_rows
        if not positives:
            self.terms = set()
            self.uncovered_positives = set()
            return

        candidates = {}
        for row in positives:
            for size in range(1, self.n_features + 1):
                for cols in combinations(range(self.n_features), size):
                    values = tuple(row[col] for col in cols)
                    term = (tuple(cols), values)
                    if term in candidates:
                        continue
                    if any(self._term_matches(term, negative) for negative in negatives):
                        continue
                    coverage = {
                        i for i, positive in enumerate(positives)
                        if self._term_matches(term, positive)
                    }
                    if coverage:
                        candidates[term] = coverage

        uncovered = set(range(len(positives)))
        selected = set()
        while uncovered:
            best_term = None
            best_gain = 0
            for term, coverage in candidates.items():
                gain = len(coverage & uncovered)
                if gain == 0:
                    continue
                if (gain > best_gain or
                    (gain == best_gain and best_term is not None and
                     (len(term[0]), term[0], tuple(map(str, term[1]))) <
                     (len(best_term[0]), best_term[0],
                      tuple(map(str, best_term[1]))))):
                    best_term = term
                    best_gain = gain
            if best_term is None:
                break
            selected.add(best_term)
            uncovered -= candidates[best_term]

        self.terms = selected
        self.uncovered_positives = {positives[i] for i in uncovered}

    def fit(self, x, y):
        x = np.asarray(x, dtype=object)
        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional feature matrix")
        self.n_features = x.shape[1]
        y = np.asarray(y).astype(int)
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of rows")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("Labels must be binary (0/1)")

        delta = {}
        for row, label in zip(x, y):
            key = tuple(row)
            delta[key] = delta.get(key, 0) + (1 if label == 1 else -1)

        self.positive_rows = {row for row, value in delta.items() if value >= 0}
        self.negative_rows = {row for row, value in delta.items() if value < 0}
        self._rebuild_formula()
        return self

    def score(self, x):
        if self.n_features is None:
            raise RuntimeError("The learner must be fitted before scoring")
        x = np.asarray(x, dtype=object).reshape(-1, self.n_features)
        return np.asarray([
            sum(self._term_matches(term, tuple(row)) for term in self.terms)
            for row in x
        ], dtype=float)

    def predict(self, x):
        return (self.score(x) > 0).astype(int)
