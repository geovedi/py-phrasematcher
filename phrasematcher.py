# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import io
import re
import vedis
import xxhash

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
        self.vocab = vedis.Vedis('{}/vocab'.format(self.model_dir))
        self.bos = vedis.Vedis('{}/bos'.format(self.model_dir))
        self.eos = vedis.Vedis('{}/eos'.format(self.model_dir))
        self.hashtable = vedis.Vedis('{}/hashtable'.format(self.model_dir))
        if pattern_file:
            if vocab_file:
                self._read_vocab(vocab_file)
            else:
                self._build_vocab(pattern_file)
            self._compile(pattern_file, max_len=max_len)
        else:
            with io.open('{}/lengths'.format(self.model_dir), 'r') as fin:
                self.lengths = set(map(int, fin.read().split()))

    def _read_vocab(self, fname):
        logging.info('Reading vocab file...')
        from collections import defaultdict
        wc = defaultdict(int)
        for line in io.open(fname, 'r', encoding='utf-8'):
            line = line.encode('utf-8')
            parts = self.tokenizer(line.lower().strip())
            word = parts[0]
            if word not in wc:
                wc[b'v:{}'.format(word)] = str(len(wc))
        n_vocab = len(wc)
        with self.vocab.transaction():
            self.vocab.mset(dict(wc))
        logging.info('Vocab size: {}'.format(n_vocab))

    def _build_vocab(self, fname):
        logging.info('Start building vocab...')
        from collections import defaultdict
        wc = defaultdict(int)
        for line in io.open(fname, 'r', encoding='utf-8'):
            line = line.encode('utf-8')
            for token in self.tokenizer(line.strip()):
                wc[b'v:{}'.format(token.lower())] += 1
        wc = sorted(wc.items(), key=lambda x: x[1])
        wc = dict((v, str(k))
                  for k, v in enumerate(reversed([k for k, v in wc])))
        n_vocab = len(wc)
        with self.vocab.transaction():
            self.vocab.mset(dict(wc))
        logging.info('Vocab size: {}'.format(n_vocab))

    def _compile(self, fname, max_len=10):
        logging.info('Start compiling patterns...')
        self.lengths = set()
        n_patterns = 0
        with self.bos.transaction(), self.eos.transaction(), \
             self.hashtable.transaction():
            for i, line in enumerate(io.open(fname, 'r', encoding='utf-8')):
                if i % 10000 == 0:
                    logging.info('Processing patterns: {}'.format(i))
                if isinstance(line, unicode):
                    line = line.encode('utf-8')
                arr = line.strip().split()
                arr = self.vocab.mget([b'v:{}'.format(t) for t in arr])
                if None in set(arr):
                    continue
                arr = [int(a) for a in arr]
                arr_len = len(arr)
                if arr_len > max_len:
                    continue
                self.bos[b'b:{}:{}'.format(arr_len, arr[0])] = b'1'
                self.eos[b'e:{}:{}'.format(arr_len, arr[-1])] = b'1'
                self.hashtable[self.hash(arr)] = b'1'
                self.lengths.add(arr_len)
                n_patterns += 1
        logging.info('Patterns: {} (Read: {})'.format(n_patterns, i + 1))
        with io.open('{}/lengths'.format(self.model_dir), 'w') as fout:
            fout.write(' '.join(str(i) for i in self.lengths))

    def hash(self, arr):
        s = b':'.join([str(i) for i in arr])
        return 'h:' + xxhash.xxh32(s).hexdigest()

    def match(self, sentence, remove_subset=False):
        if isinstance(sentence, unicode):
            sentence = sentence.encode('utf-8')
        tok = self.tokenizer(sentence.strip())
        tok_arr = self.vocab.mget([b'v:{}'.format(t) for t in tok])
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
