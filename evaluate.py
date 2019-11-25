import argparse
import log
import os
import collections
import time
from collections import defaultdict
from typing import List, Dict

import jsonpickle

import utils
from dataset import file_to_dataset, DatasetEntry, SYNONYM, RELATIONS
from patterns import get_patterns
from mlm import BertMaskedLanguageModel, AbstractMaskedLanguageModel, RobertaMaskedLanguageModel

logger = log.get_logger('root')

MODELS = {
    'bert': BertMaskedLanguageModel,
    'roberta': RobertaMaskedLanguageModel,
}


class EntryResult:
    def __init__(self, entry: DatasetEntry, predictions: List[List[str]], rank: int, precision_at: Dict[int, float]):
        self.entry = entry
        self.predictions = predictions
        self.rank = rank
        self.precision_at = precision_at

    def to_tsv_str(self, use_rank: bool, precision_k_values: List[int]) -> str:
        predictions_str = [','.join(pred) for pred in self.predictions]
        predictions_str = ' '.join(predictions_str)

        ret = self.entry.base_word.word + '\t' \
              + self.entry.relation + '\t' \
              + ' '.join([w.word for w in self.entry.matching_words])
        if use_rank:
            ret += '\t{}'.format(self.rank)
        for k in precision_k_values:
            ret += '\t{}'.format(self.precision_at[k])
        ret += '\t' + predictions_str
        return ret

    @staticmethod
    def headline_tsv(use_rank: bool, precision_k_values: List[int]):
        headline = 'keyword\trelation\ttargets'
        if use_rank:
            headline += '\tMRR'
        for k in precision_k_values:
            headline += '\tP@{}'.format(k)
        headline += '\tpredictions'
        return headline

    @staticmethod
    def to_file(results: List['EntryResult'], use_rank: bool, precision_k_values: List[int], path: str) -> None:
        with open(path, 'w', encoding='utf8') as f:
            f.write(EntryResult.headline_tsv(use_rank, precision_k_values) + '\n')
            for res in results:
                f.write(res.to_tsv_str(use_rank, precision_k_values) + '\n')


class Result:
    def __init__(self, mrr: float, precision_at: Dict[int, float], ranks: List[float] = None,
                 precision_vals: Dict[int, List[float]] = None, entry_results: List[EntryResult] = None):
        self.mrr = mrr
        self.precision_at = precision_at
        self.ranks = ranks
        self.precision_vals = precision_vals
        self.entry_results = entry_results

    def stringify(self, use_mrr, precision_k_values):
        ret = ''
        if use_mrr:
            ret += '{:5.3f} '.format(self.mrr)
        for k in precision_k_values:
            ret += '{:5.3f} '.format(self.precision_at[k])
        return ret

    @staticmethod
    def stringify_results(results: Dict[str, 'Result'], use_mrr=True, precision_k_values=None) -> str:
        space_for_name = max(len(key) for key in results.keys()) + 2
        ret = ' ' * space_for_name + Result.headline(use_mrr, precision_k_values) + '\n'

        for key in results:
            ret += (('{:' + str(space_for_name) + 's}').format(key) +
                    results[key].stringify(use_mrr, precision_k_values)) + '\n'
        return ret

    @staticmethod
    def headline(use_mrr, precision_k_values) -> str:
        headline = ''
        if use_mrr:
            headline += 'MRR   '
        for k in precision_k_values:
            headline += 'P@{:<3d} '.format(k)
        return headline


def evaluate_from_predictions(dataset: List[DatasetEntry], predictions: Dict[str, List[List[str]]], compute_mrr=True,
                              precision_at=None, with_raw: bool = False) -> Result:
    if precision_at is None:
        precision_at = [3, 10, 100]

    reciprocal_ranks = []
    precision_vals = defaultdict(list)
    entry_results = []

    for idx, entry in enumerate(dataset):

        if entry.id not in predictions:
            logger.warning('Found no predictions for entry with id "{}"'.format(entry.id))
            continue

        prediction = predictions[entry.id]
        actuals = [w.word for w in entry.matching_words]

        entry_rank = -1
        entry_precision_at = {}

        if compute_mrr:
            reciprocal_rank = get_reciprocal_rank(actuals, prediction)
            reciprocal_ranks.append(reciprocal_rank)
            entry_rank = 1 / reciprocal_rank if reciprocal_rank > 0 else 1000

        for k in precision_at:
            precision_at_k = get_precision_at(k, actuals, prediction)
            precision_vals[k].append(precision_at_k)
            entry_precision_at[k] = precision_at_k

        entry_result = EntryResult(entry, prediction, entry_rank, entry_precision_at)
        entry_results.append(entry_result)

        if idx % 100 == 0:
            logger.info('Done processing {} of {} entries'.format(idx + 1, len(dataset)))

    result = Result(0, {}, entry_results=entry_results)

    if compute_mrr:
        mrr = avg(reciprocal_ranks)
        result.mrr = mrr
    for k in precision_at:
        p_at_k = avg(precision_vals[k])
        result.precision_at[k] = p_at_k

    if with_raw:
        result.ranks = [1 / r if r > 0 else 1000 for r in reciprocal_ranks]
        result.precision_vals = precision_vals
    return result


def avg(l: List[float]):
    if len(l) == 0:
        logger.warning('Computing average of empty list, returning -1 instead')
        return -1
    return sum(l) / len(l)


def get_reciprocal_rank(actuals: List[str], predictions: List[List[str]]):
    rr = 0
    for pattern_predictions in predictions:
        for idx, predicted_word in enumerate(pattern_predictions):
            if predicted_word in actuals:
                rr = max(1 / (idx + 1), rr)
    return rr


def get_precision_at(k: int, actuals: List[str], predictions: List[List[str]]):
    matches = get_matches_at(k, actuals, predictions)
    return min(1, len(matches) / k)


def get_matches_at(k: int, actuals: List[str], predictions: List[List[str]], use_sum_instead_of_max: bool = False):
    if use_sum_instead_of_max:
        all_predictions = set()
        for pattern_predictions in predictions:
            all_predictions.update(pattern_predictions[:k])
        return get_matches(all_predictions, actuals)
    else:
        best_predictions = set()
        for pattern_predictions in predictions:
            pp_set = set(pattern_predictions[:k])
            if len(get_matches(pp_set, actuals)) > len(get_matches(best_predictions, actuals)):
                best_predictions = pp_set
        return get_matches(best_predictions, actuals)


def predictions_to_file(model: AbstractMaskedLanguageModel, dataset: List[DatasetEntry], num_predictions=100,
                        out_path: str = None) -> None:
    if os.path.isfile(out_path):
        raise FileExistsError('File {} already exists'.format(out_path))

    with open(out_path, 'w', encoding='utf-8') as out_file:

        t0 = time.time()

        for idx, entry in enumerate(dataset):
            if entry.relation == SYNONYM:
                continue

            entry_predictions = predictions_for_entry(model, entry, num_predictions)
            _write_predictions_to_file(out_file, entry.id, entry_predictions)

            if idx % 100 == 0:
                total_time = time.time() - t0
                time_per_entry = total_time / (idx + 1)
                remaining_entries = len(dataset) - (idx + 1)
                time_for_remaining_entries = remaining_entries * time_per_entry
                logger.info('Done processing {} of {} dataset entries, estimated remaining time: {}s'.format(
                    idx + 1, len(dataset), time_for_remaining_entries))


def _write_predictions_to_file(file, entry_id, predictions: List[List[str]]) -> None:
    file.write(str(entry_id) + '\t' + jsonpickle.dumps(predictions) + '\n')


def _load_predictions_from_file(path: str) -> Dict[str, List[List[str]]]:
    logger.info("Loading model predictions from {}".format(path))
    predictions = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.split('\t', 1)
            value_as_list = jsonpickle.decode(value)
            predictions[key] = value_as_list
    logger.info('Done loading model predictions')
    return predictions


def predictions_for_entry(model: AbstractMaskedLanguageModel, entry: DatasetEntry, num_predictions=100):
    # get the corresponding patterns
    patterns = get_patterns(entry.base_word, entry.relation)
    predictions = []

    for pattern in patterns:
        pattern_predictions = model.get_predictions(pattern, entry.base_word.word, num_predictions)
        predictions.append(pattern_predictions)

    return predictions


def get_matches(predictions, actuals):
    return predictions.intersection(actuals)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # file parameters
    parser.add_argument('--root', type=str, default='C:/Users/Timo/Desktop/bert-experiments/')
    parser.add_argument('--dataset', type=str, default='eval-dataset.txt')
    parser.add_argument('--predictions_file', type=str, default=None, required=True)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--raw_output_file', default=None, type=str)

    # parameters for computing new predictions
    parser.add_argument('--model_cls', choices=['bert', 'roberta'], default='bert')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--num_predictions', type=int, default=100)

    # evaluation parameters
    parser.add_argument('--print_statistics', action='store_true',
                        help='If set, instead of evaluating a model, statistics about the used dataset are printed.')
    parser.add_argument('--set_type', choices=['dev', 'test'], default=None)
    parser.add_argument('--count_thresholds', '-cs', type=int, nargs='*', default=[10, 100])
    parser.add_argument('--min_subset_size', type=int, default=10)
    parser.add_argument('--precision_at', type=int, nargs='*', default=[3, 10, 100])
    parser.add_argument('--keep_frequent_corruptions', action='store_true',
                        help='If set, corruption entries with key word frequencies at or above 10 are kept')

    args = parser.parse_args()
    ds = file_to_dataset(os.path.join(args.root, args.dataset),
                         keep_frequent_corruptions=args.keep_frequent_corruptions)

    if args.print_statistics:
        ds.print_statistics()

    predictions_file = os.path.join(args.root, args.predictions_file)

    if os.path.isfile(predictions_file):
        predictions = _load_predictions_from_file(predictions_file)

    else:
        logger.info('Found no precomputed predictions at {}'.format(predictions_file))
        embeddings = None

        if args.embeddings:
            embeddings = utils.load_embeddings(os.path.join(args.root, args.embeddings))

        model_cls = MODELS[args.model_cls]
        model = model_cls(args.model_name, embeddings)
        predictions_to_file(model, ds, args.num_predictions, predictions_file)
        predictions = _load_predictions_from_file(predictions_file)

    result = evaluate_from_predictions(ds.select(set_type=args.set_type), predictions)
    result_dict = collections.OrderedDict()
    result_dict['all_values'] = result

    count_thresholds = [0] + args.count_thresholds + [-1]

    for lower_bound, upper_bound in utils.pairwise(count_thresholds):
        ds_restricted = ds.select(count=(lower_bound, upper_bound), set_type=args.set_type)

        if len(ds_restricted) >= args.min_subset_size:
            result = evaluate_from_predictions(ds_restricted, predictions)
            result_dict['all_values ({},{})'.format(lower_bound, upper_bound)] = result

            if args.raw_output_file:
                EntryResult.to_file(result.entry_results, True, args.precision_at,
                                    os.path.join(args.root,
                                                 args.raw_output_file + '-{}-{}'.format(lower_bound, upper_bound)))

    for rel in RELATIONS:
        for lower_bound, upper_bound in utils.pairwise(count_thresholds):
            ds_restricted = ds.select(relation=rel, count=(lower_bound, upper_bound), set_type=args.set_type)

            if len(ds_restricted) >= args.min_subset_size:
                result = evaluate_from_predictions(ds_restricted, predictions)
                result_dict['{} ({},{})'.format(rel, lower_bound, upper_bound)] = result

    results_str = Result.stringify_results(result_dict, True, args.precision_at)
    print(results_str)
    if args.output_file:
        with open(os.path.join(args.root, args.output_file), 'w', encoding='utf8') as f:
            f.write(results_str)

    if args.raw_output_file:
        EntryResult.to_file(result_dict['all_values'].entry_results, True, args.precision_at,
                            os.path.join(args.root, args.raw_output_file))
