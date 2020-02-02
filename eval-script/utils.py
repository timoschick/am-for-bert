from typing import Dict

import itertools
import torch
import io

import log

logger = log.get_logger('root')


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def load_embeddings(embd_file: str) -> Dict[str, torch.Tensor]:
    logger.info('Loading embeddings from {}'.format(embd_file))
    embds = {}
    with io.open(embd_file, 'r', encoding='utf8') as f:
        for line in f:
            comps = line.split()
            word = comps[0]
            embd = [float(x) for x in comps[1:]]
            embds[word] = torch.tensor(embd)
    logger.info('Found {} embeddings'.format(len(embds)))
    return embds
