import statistics
from typing import Tuple, List
import log

logger = log.get_logger('root')

SYNONYM = 'synonym'
ANTONYM = 'antonym'
HYPERNYM = 'hypernym'
COHYPONYM = 'cohyponym'
CORRUPTION = 'corruption'

TEST = 'test'
DEV = 'dev'

RELATIONS = [SYNONYM, ANTONYM, HYPERNYM, COHYPONYM, CORRUPTION]


class AnnotatedWord:
    def __init__(self, word, pos=None, freq=None, count=None):
        self.word = word
        self.pos = pos
        self.freq = freq
        self.count = count

    def __repr__(self):
        return '{} ({},{},{})'.format(self.word, self.pos, self.freq, self.count)


class DatasetEntry:
    def __init__(self, base_word: AnnotatedWord, relation: str, matching_words: List[AnnotatedWord],
                 set_type: str = TEST, eid=None):
        if relation not in RELATIONS:
            raise ValueError('Relation must be one of {}, got {}'.format(RELATIONS, relation))
        self.base_word = base_word
        self.relation = relation
        self.matching_words = matching_words
        self.set_type = set_type
        self.id = eid

    def __repr__(self):
        return '{}({}): {} <{}> {}'.format(self.id, self.set_type, self.base_word, self.relation, self.matching_words)


class Dataset(list):

    def select(self, pos: str = None, relation: str = None, freq: Tuple[int, int] = None,
               count: Tuple[int, int] = None, set_type: str = None) -> 'Dataset':
        ret = Dataset()
        for entry in self:
            if pos and entry.base_word.pos != pos:
                continue
            if relation and entry.relation != relation:
                continue
            if freq:
                if entry.base_word.freq < freq[0]:
                    continue
                if 0 < freq[1] <= entry.base_word.freq:
                    continue
            if count:
                if entry.base_word.count < count[0]:
                    continue
                if 0 < count[1] <= entry.base_word.count:
                    continue
            if set_type and entry.set_type != set_type:
                continue
            ret.append(entry)

        return ret

    def print_statistics(self, relations=None, set_types=None, counts=None) -> None:
        if relations is None:
            relations = RELATIONS

        if set_types is None:
            set_types = [DEV, TEST]

        if counts is None:
            counts = [(0, 9), (10, 99), (100, -1)]

        for relation in relations:
            for set_type in set_types:

                for (min_count, max_count) in counts:
                    ds_subset = self.select(relation=relation, set_type=set_type, count=(min_count, max_count))

                    if not ds_subset:
                        continue
                    matching_words = [len(x.matching_words) for x in ds_subset]

                    logger.info('{} - {} ({},{}): size = {}, mean targets = {}, median targets = {}'.format(
                        relation, set_type, min_count, max_count, len(ds_subset),
                        statistics.mean(matching_words), statistics.median(matching_words)
                    ))


def _string_to_entry(estr: str) -> DatasetEntry:
    cmps = estr.split('\t')
    eid = cmps[0]
    set_type = cmps[1]
    base_word = _string_to_annotated_word(cmps[2])
    relation = cmps[3]
    matching_words = [_string_to_annotated_word(w) for w in cmps[4:]]
    return DatasetEntry(base_word, relation, matching_words, eid=eid, set_type=set_type)


def _string_to_annotated_word(astr: str) -> AnnotatedWord:
    # e.g. disease (n,4.95,64105)
    word, meta_info = astr.split()
    meta_info = meta_info[1:-1]
    pos, freq, count = meta_info.split(',')
    freq = float(freq)
    count = int(count)
    return AnnotatedWord(word, pos=pos, freq=freq, count=count)


def file_to_dataset(path: str, keep_frequent_corruptions: bool = False) -> Dataset:
    logger.info('Loading dataset from {}'.format(path))
    ret = Dataset()
    with open(path, 'r', encoding='utf8') as file:
        for line in file:
            entry = _string_to_entry(line)
            if entry.relation == CORRUPTION and entry.base_word.count >= 10 and not keep_frequent_corruptions:
                continue
            ret.append(entry)
    logger.info('Done loading dataset')
    return ret
