# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import io
import re
import binascii
import pickle

from collections import defaultdict

import logging
logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


class Patterns(object):
    __slots__ = ('p_len', 'p_obj')

    def __init__(self):
        self.p_len = set()
        self.p_obj = set()


class PhraseMatcher(object):
    def __init__(self,
                 model_dir,
                 pattern_file=None,
                 vocab_file=None,
                 max_len=10,
                 tokenizer=lambda x: x.split()):
        self.tokenizer = tokenizer
        self.model_dir = model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if pattern_file:
            if vocab_file:
                self._read_vocab(vocab_file)
            else:
                self._build_vocab(pattern_file)

            self._compile(pattern_file, max_len=max_len)
        else:
            fin = open('{}/vocab.p'.format(self.model_dir), 'rb')
            self.vocab = pickle.load(fin)

            fin = open('{}/patterns.p'.format(self.model_dir), 'rb')
            self.patterns = pickle.load(fin)

    def _read_vocab(self, fname):
        logging.info('Reading vocab file...')
        wc = defaultdict(int)

        for line in io.open(fname, 'r', encoding='utf-8'):
            parts = self.tokenizer(line.lower().strip())
            word = parts[0]
            if word not in wc:
                wc[word] = len(wc)

        n_vocab = len(wc)
        self.vocab = dict(wc)

        with open('{}/vocab.p'.format(self.model_dir), 'wb') as fout:
            pickle.dump(self.vocab, fout, -1)
        logging.info('Vocab size: {}'.format(n_vocab))

    def _build_vocab(self, fname):
        logging.info('Start building vocab...')
        wc = defaultdict(int)

        for line in io.open(fname, 'r', encoding='utf-8'):
            for word in self.tokenizer(line.lower().strip()):
                wc[word] += 1

        wc = sorted(wc.items(), key=lambda x: x[1])
        wc = dict((v, k) for k, v in enumerate(reversed([k for k, v in wc])))
        n_vocab = len(wc)
        self.vocab = wc

        with open('{}/vocab.p'.format(self.model_dir), 'wb') as fout:
            pickle.dump(self.vocab, fout, -1)
        logging.info('Vocab size: {}'.format(n_vocab))

    def _compile(self, fname, max_len=10):
        logging.info('Start compiling patterns...')
        self.patterns = Patterns()

        for i, pat in enumerate(io.open(fname, 'r', encoding='utf-8')):
            if i % 100000 == 0:
                logging.info('Processing input patterns: {}'.format(i))

            p_arr = pat.strip().split()
            p_len = len(p_arr)

            if p_len > max_len:
                continue

            p_ints = [self.vocab.get(t, None) for t in p_arr]
            if None in set(p_ints):
                continue

            p_hash = self.hash(' '.join(p_arr))
            p_obj = (p_len, p_ints[0], p_ints[-1], p_hash)

            self.patterns.p_obj.add(p_obj)
            self.patterns.p_len.add(p_len)

        with open('{}/patterns.p'.format(self.model_dir), 'wb') as fout:
            pickle.dump(self.patterns, fout, -1)

    def hash(self, text):
        s = text.encode('utf-8')
        return binascii.crc32(s) % (1 << 32)

    def match(self, sentence, remove_subset=False):
        tok = self.tokenizer(sentence.strip())
        tok_ints = [self.vocab.get(t, None) for t in tok]
        tok_len = len(tok_ints)
        candidates = set()

        for i, b_int in enumerate(tok_ints):
            if b_int == None:
                continue

            for p_len in self.patterns.p_len:
                j = i + p_len - 1
                if j + 1 > tok_len:
                    continue

                p_ints = tok_ints[i:j + 1]
                if None in set(p_ints):
                    continue

                e_int = tok_ints[j]
                if e_int == None:
                    continue

                p_hash = self.hash(' '.join(tok[i:j + 1]))
                p_obj = (p_len, b_int, e_int, p_hash)

                if p_obj in self.patterns.p_obj:
                    candidates.add((i, j))

        if remove_subset:
            ranges = list(sorted(candidates, reverse=True))

            for (i, j) in list(candidates):
                for ii, jj in ranges:
                    if i == ii and j == jj:
                        continue

                    if ii <= i and j <= jj:
                        try:
                            candidates.remove((i, j))
                        except KeyError:
                            pass

        for (i, j) in candidates:
            yield tok[i:j + 1]

