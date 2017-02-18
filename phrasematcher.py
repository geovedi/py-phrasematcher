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
            (self.vocab, self.bos, self.eos, self.lengths) = pickle.load(fin)
        except:
            self.vocab = {}
            self.bos, self.eos, self.lengths = set(), set(), set()
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
            pickle.dump((self.vocab, self.bos, self.eos, self.lengths), fout)
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
            pickle.dump((self.vocab, self.bos, self.eos, self.lengths), fout)
        logging.info('Vocab size: {}'.format(n_vocab))

    def _compile(self, fname, max_len=10):
        logging.info('Start compiling patterns...')
        n_patterns = 0
        hash_buf = {}
        self.hashtable.begin()
        for i, line in enumerate(io.open(fname, 'r', encoding='utf-8')):
            if i % 100000 == 0:
                logging.info('Processing patterns: {}'.format(i))
                self.hashtable.mset(hash_buf)
                hash_buf = {}
                self.hashtable.commit()
            arr = line.strip().split()
            arr = [self.vocab.get(t, None) for t in arr]
            arr_len = len(arr)
            if None in set(arr) or arr_len > max_len:
                continue
            self.bos.add(b'b:{}:{}'.format(arr_len, arr[0]))
            self.eos.add(b'e:{}:{}'.format(arr_len, arr[-1]))
            hash_buf[self.hash(arr)] = b'1'
            self.lengths.add(arr_len)
            n_patterns += 1
        self.hashtable.mset(hash_buf)
        self.hashtable.commit()
        logging.info('Patterns: {} (Read: {})'.format(n_patterns, i + 1))
        with open('{}/model'.format(self.model_dir), 'wb') as fout:
            pickle.dump((self.vocab, self.bos, self.eos, self.lengths), fout)

    def hash(self, arr):
        s = b':'.join([str(i) for i in arr])
        return 'h:' + xxhash.xxh32(s).hexdigest()

    def match(self, sentence, remove_subset=False):
        tok = self.tokenizer(sentence.strip())
        tok_arr = [self.vocab.get(t, None) for t in tok]
        tok_len = len(tok_arr)
        candidates = set()
        for i, tok_int in enumerate(tok_arr):
            if tok_int == None:
                continue
            for p_len in self.lengths:
                b_int = int(tok_int)
                j = i + p_len
                if j + 1 <= tok_len:
                    p_arr = tok_arr[i:j]
                    if None in set(p_arr):
                        continue
                    e_int = int(tok_arr[j - 1])
                    b_idx = b'b:{}:{}'.format(p_len, b_int)
                    e_idx = b'e:{}:{}'.format(p_len, e_int)
                    if b_idx in self.bos and e_idx in self.eos:
                        if self.hash(p_arr) in self.hashtable:
                            candidates.add((i, j))
        if remove_subset:
            ranges = list(sorted(candidates, reverse=True))
            for (i, j), c_idx in list(candidates):
                for ii, jj in ranges:
                    if i == ii and j == jj:
                        continue
                    if ii <= i and j <= jj:
                        candidates.remove((i, j))
        for (i, j) in candidates:
            yield tok[i:j]
