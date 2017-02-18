# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import io
import re
import pickle
import vedis
import xxhash
from collections import defaultdict

import logging
logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


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

        try:
            fin = open('{}/model'.format(self.model_dir), 'rb')
            (self.vocab, self.b, self.e, self.lengths) = pickle.load(fin)
        except:
            self.vocab = {}
            self.b, self.e, self.lengths = set(), set(), set()

        self.hashtable = vedis.Vedis('{}/hashtable'.format(self.model_dir))

        if pattern_file:
            if vocab_file:
                self._read_vocab(vocab_file)
            else:
                self._build_vocab(pattern_file)

            self._compile(pattern_file, max_len=max_len)

    def _read_vocab(self, fname):
        logging.info('Reading vocab file...')
        wc = defaultdict(int)

        for line in io.open(fname, 'r', encoding='utf-8'):
            parts = self.tokenizer(line.lower().strip())
            word = parts[0]
            if word not in wc:
                wc[word] = len(wc)

        n_vocab = len(wc)
        self.vocab = wc

        with open('{}/model'.format(self.model_dir), 'wb') as fout:
            pickle.dump((self.vocab, self.b, self.e, self.lengths), fout)
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

        with open('{}/model'.format(self.model_dir), 'wb') as fout:
            pickle.dump((self.vocab, self.b, self.e, self.lengths), fout)
        logging.info('Vocab size: {}'.format(n_vocab))

    def _compile(self, fname, max_len=10):
        logging.info('Start compiling patterns...')
        n_patterns = 0
        hash_buf = {}
        self.hashtable.begin()

        for i, line in enumerate(io.open(fname, 'r', encoding='utf-8')):
            if i % 10000 == 0:
                logging.info('Processing patterns: {}'.format(i))

            p_arr = line.strip().split()
            p_len = len(p_arr)

            if p_len > max_len:
                continue

            p_arr = [self.vocab.get(t, None) for t in p_arr]
            if None in set(p_arr):
                continue

            b_idx = '{}:{}'.format(p_len, p_arr[0])
            e_idx = '{}:{}'.format(p_len, p_arr[-1])
            self.b.add(b_idx)
            self.e.add(e_idx)

            p_hash = self.hash(p_arr)
            hash_buf[p_hash] = b'1'

            self.lengths.add(p_len)
            n_patterns += 1

            if n_patterns % 100000 == 0:
                self.hashtable.mset(hash_buf)
                self.hashtable.commit()
                hash_buf = {}

        self.hashtable.mset(hash_buf)
        self.hashtable.commit()
        logging.info('Patterns: {} (Read: {})'.format(n_patterns, i + 1))

        with open('{}/model'.format(self.model_dir), 'wb') as fout:
            pickle.dump((self.vocab, self.b, self.e, self.lengths), fout)

    def hash(self, arr):
        s = b':'.join(['{}'.format(i) for i in arr])
        return xxhash.xxh32(s).hexdigest()

    def match(self, sentence, remove_subset=False):
        tok = self.tokenizer(sentence.strip())
        tok_arr = [self.vocab.get(t, None) for t in tok]
        tok_len = len(tok_arr)
        candidates = set()

        for i, b_tok in enumerate(tok_arr):
            if b_tok == None:
                continue

            for p_len in self.lengths:
                j = i + p_len
                if j + 1 > tok_len:
                    continue

                b_idx = '{}:{}'.format(p_len, b_tok)
                if b_idx not in self.b:
                    continue

                e_tok = tok_arr[j - 1]
                if e_tok == None:
                    continue

                e_idx = '{}:{}'.format(p_len, e_tok)
                if e_idx not in self.e:
                    continue

                p_arr = tok_arr[i:j]
                if None in set(p_arr):
                    continue

                p_hash = self.hash(p_arr)
                if p_hash in self.hashtable:
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
            yield tok[i:j]

