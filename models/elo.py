import json
from collections import defaultdict


class EloModel(object):

    DEFAULT_ELO_RATING = 1500

    def __init__(self, c=250., o=5., s=0.4, match_counts=None, beta=None):
        self.c = c
        self.o = o
        self.s = s
        if match_counts is None:
            match_counts = {}
        self.match_counts = defaultdict(lambda: 0, match_counts)

        if beta is None:
            beta = {}
        self.beta = defaultdict(lambda: self.DEFAULT_ELO_RATING, beta)

    @staticmethod
    def elo(x):
        # Just a rescaled sigmoid
        return 1. / (1. + 10. ** (-x / 400.))

    def update(self, p1_id, p2_id, y, weight=1., match_id=None):
        pred = self.predict(p1_id, p2_id)

        lr1 = weight * self.c / ((self.match_counts[p1_id] + self.o) ** self.s)
        lr2 = weight * self.c / ((self.match_counts[p2_id] + self.o) ** self.s)

        new_beta1 = self.beta[p1_id] + lr1 * (y - pred)
        new_beta2 = self.beta[p2_id] + lr2 * (pred - y)

        self.beta[p1_id] = new_beta1
        self.beta[p2_id] = new_beta2

        self.match_counts[p1_id] += 1
        self.match_counts[p2_id] += 1

        return match_id, new_beta1, new_beta2, pred

    def fit(self, p1_ids, p2_ids, y, weights, match_ids=None):
        if match_ids is None:
            match_ids = [None] * len(p1_ids)
        for p1_id, p2_id, y, w, match_id in zip(
            p1_ids,
            p2_ids,
            y,
            weights,
            match_ids
        ):
            self.update(p1_id, p2_id, y, weight=w, match_id=match_id)

    def predict(self, p1_id, p2_id):
        return self.elo(self.beta[p1_id] - self.beta[p2_id])

    def state_to_dict(self):
        return json.dumps({
            'beta': self.beta,
            'match_counts': self.match_counts,
            'c': self.c,
            'o': self.o,
            's': self.s
        })
