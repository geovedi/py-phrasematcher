# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import io
import re
from collections import defaultdict


class PhraseMatcher(object):
    def __init__(self, pattern_file, tokenizer=lambda x: x.split()):
        self.tokenizer = tokenizer
        self._build_vocab(pattern_file)
        self._compile(pattern_file)

    def _build_vocab(self, fname):
        vocab = defaultdict(int)
        for line in io.open(fname, 'r', encoding='utf-8'):
            for token in self.tokenizer(line.strip()):
                vocab[token.lower()] += 1
        vocab = sorted(vocab.items(), key=lambda x: x[1])
        vocab = dict((v, k + 1)
                     for k, v in enumerate(reversed([k for k, v in vocab])))
        vocab['<unk>'] = 0
        self.vocab = vocab

    def _compile(self, fname):
        self.bos = defaultdict(set)
        self.eos = defaultdict(set)
        self.checksums = defaultdict(set)
        self.lengths = set()

        for line in io.open(fname, 'r', encoding='utf-8'):
            arr = [self.vocab.get(t.lower(), 0) for t in line.strip().split()]
            if 0 in arr:
                continue
            arr_len = len(arr)
            self.bos[arr_len].add(arr[0])
            self.eos[arr_len].add(arr[-1])
            self.checksums[(arr_len, arr[0], arr[-1])].add(self.checksum(arr))
            self.lengths.add(arr_len)

    def checksum(self, arr):
        """fletcher checksum"""
        sum1 = sum2 = 0
        for v in arr:
            sum1 = (sum1 + v) % 255
            sum2 = (sum1 + sum1) % 255
        return (sum1 * 256) + sum2

    def match(self, sentence, remove_subset=False):
        tok = self.tokenizer(sentence.strip())
        tok_arr = [self.vocab.get(t.lower(), 0) for t in tok]
        tok_len = len(tok_arr)
        candidates = set()
        ranges = set()

        # find markers
        for i, tok_int in enumerate(tok_arr):
            if tok_int == 0:
                continue
            for p_len in self.lengths:
                b_int = tok_int
                j = i + p_len
                if j + 1 <= tok_len:
                    e_int = tok_arr[j - 1]
                    if e_int == 0:
                        continue
                    if b_int in self.bos[p_len] and e_int in self.eos[p_len]:
                        c_idx = (p_len, b_int, e_int)
                        candidates.add(((i, j), c_idx))
                        ranges.add((i, j))

        if remove_subset:
            # filter out overlapping candidates
            ranges = list(sorted(ranges, reverse=True))
            for (i, j), c_idx in list(candidates):
                for ii, jj in ranges:
                    if i == ii and j == jj:
                        continue
                    if ii <= i and j <= jj:
                        candidates.remove(((i, j), c_idx))

        # check candidates
        for (i, j), c_idx in candidates:
            checksums = self.checksums.get(c_idx)
            if not checksums:
                continue
            checksum = self.checksum(tok_arr[i:j])
            if checksum in checksums:
                yield tok[i:j]
