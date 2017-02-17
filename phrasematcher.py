# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import io
import re
import binascii
from collections import defaultdict
import logging
logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

UNK = 0


class PhraseMatcher(object):
    def __init__(self, pattern_file, vocab_file=None, max_len=10,
                 tokenizer=lambda x: x.split()):
        self.tokenizer = tokenizer
        if vocab_file:
            self._read_vocab(vocab_file)
        else:
            self._build_vocab(pattern_file)
        self._compile(pattern_file, max_len=max_len)

    def _read_vocab(self, fname):
        logging.info('Reading vocab file...')
        vocab = defaultdict(int)
        vocab['<unk>'] = UNK
        for line in io.open(fname, 'r', encoding='utf-8'):
            parts = self.tokenizer(line.lower().strip())
            if len(parts) == 2:
                if word not in vocab:
                    vocab[word] = len(vocab)
        self.vocab = vocab
        logging.info('Vocab size: {}'.format(len(self.vocab)))

    def _build_vocab(self, fname):
        logging.info('Start building vocab...')
        vocab = defaultdict(int)
        for line in io.open(fname, 'r', encoding='utf-8'):
            for token in self.tokenizer(line.strip()):
                vocab[token.lower()] += 1
        vocab = sorted(vocab.items(), key=lambda x: x[1])
        vocab = dict((v, k + 1)
                     for k, v in enumerate(reversed([k for k, v in vocab])))
        vocab['<unk>'] = UNK
        self.vocab = vocab
        logging.info('Vocab size: {}'.format(len(self.vocab)))

    def _compile(self, fname, max_len=10):
        logging.info('Start compiling patterns...')
        self.bos = defaultdict(set)
        self.eos = defaultdict(set)
        self.checksums = defaultdict(set)
        self.lengths = set()
        n_patterns = 0

        for line_no, line in enumerate(io.open(fname, 'r', encoding='utf-8')):
            if line_no % 10000 == 0:
                logging.info('Processing patterns: {}'.format(line_no))
            arr = [self.vocab.get(t.lower(), 0) for t in line.strip().split()]
            if UNK in set(arr):
                continue
            arr_len = len(arr)
            if arr_len > max_len:
                continue
            self.bos[arr_len].add(arr[0])
            self.eos[arr_len].add(arr[-1])
            c_arr = self.checksum(arr)
            c_idx = (arr_len, arr[0], arr[-1])
            self.checksums[c_arr].add(c_idx)
            self.lengths.add(arr_len)
            n_patterns += 1
        logging.info('Patterns: {} (orig: {})'.format(n_patterns, line_no))

    def checksum(self, arr):
        return binascii.crc32(str(arr))

    def match(self, sentence, remove_subset=False):
        tok = self.tokenizer(sentence.strip())
        tok_arr = [self.vocab.get(t.lower(), 0) for t in tok]
        tok_len = len(tok_arr)
        candidates = set()
        ranges = set()

        for i, tok_int in enumerate(tok_arr):
            if tok_int == UNK:
                continue
            for p_len in self.lengths:
                b_int = tok_int
                j = i + p_len
                if j + 1 <= tok_len:
                    e_int = tok_arr[j - 1]
                    p_arr = tok_arr[i:j]
                    if UNK in set(p_arr):
                        continue
                    if b_int in self.bos[p_len] and e_int in self.eos[p_len]:
                        c_idx = (p_len, b_int, e_int)
                        c_arr = self.checksum(p_arr)
                        if c_idx in self.checksums.get(p_arr, set()):
                            candidates.add(((i, j), c_idx))
                            ranges.add((i, j))

        if remove_subset:
            ranges = list(sorted(ranges, reverse=True))
            for (i, j), c_idx in list(candidates):
                for ii, jj in ranges:
                    if i == ii and j == jj:
                        continue
                    if ii <= i and j <= jj:
                        candidates.remove(((i, j), c_idx))

        for (i, j), c_idx in candidates:
            yield tok[i:j]
