""" beam-search utilities"""
from collections import Counter

from cytoolz import concat

import torch


class Beam(object):
  def __init__(self, tokens, log_probs, state, context):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context

  def extend(self, token, log_prob, state, context):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / (len(self.tokens) - 1)
