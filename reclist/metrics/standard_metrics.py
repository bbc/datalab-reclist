import os
import random
import itertools
import numpy as np
import collections
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
from bisect import bisect_right, bisect_left
from reclist.current import current


def statistics(x_train, y_train, x_test, y_test, y_pred):
    train_size = len(x_train)
    test_size = len(x_test)
    # num non-zero preds
    num_preds = len([p for p in y_pred if p])
    return {
        'training_set__size': train_size,
        'test_set_size': test_size,
        'num_non_null_predictions': num_preds
    }


def sample_hits_at_k(y_preds, y_test, x_test=None, k=3, size=3):
    hits = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_test)):
        if _y[0] in _p[:k]:
            hit_info = {
                'Y_TEST': [_y[0]],
                'Y_PRED': _p[:k],
            }
            if x_test:
                hit_info['X_TEST'] = [x_test[idx][0]]
            hits.append(hit_info)

    if len(hits) < size or size == -1:
        return hits
    return random.sample(hits, k=size)


def sample_misses_at_k(y_preds, y_test, x_test=None, k=3, size=3):
    misses = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_test)):
        if _y[0] not in _p[:k]:
            miss_info =  {
                'Y_TEST': [_y[0]],
                'Y_PRED': _p[:k],
            }
            if x_test:
                miss_info['X_TEST'] = [x_test[idx][0]]
            misses.append(miss_info)

    if len(misses) < size or size == -1:
        return misses
    return random.sample(misses, k=size)


def hit_rate_at_k_nep(y_preds, y_test, k=3):
    y_test = [[k] for k in y_test]
    return hit_rate_at_k(y_preds, y_test, k=k)


def hit_rate_at_k(y_preds, y_test, k=3):
    hits = 0
    for _p, _y in zip(y_preds, y_test):
        if len(set(_p[:k]).intersection(set(_y))) > 0:
            hits += 1
    return hits / len(y_test)


def mrr_at_k_nep(y_preds, y_test, k=3):
    """
    Computes MRR

    :param y_preds: predictions, as lists of lists
    :param y_test: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    y_test = [[k] for k in y_test]
    return mrr_at_k(y_preds, y_test, k=k)


def mrr_at_k(y_preds, y_test, k=3):
    """
    Computes MRR

    :param y_preds: predictions, as lists of lists
    :param y_test: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    rr = []
    for _p, _y in zip(y_preds, y_test):
        for rank, p in enumerate(_p[:k], start=1):
            if p in _y:
                rr.append(1 / rank)
                break
        else:
            rr.append(0)
    assert len(rr) == len(y_preds)
    return np.mean(rr)


def coverage_at_k(y_preds, product_data, k=3):
    pred_skus = set(itertools.chain.from_iterable(y_preds[:k]))
    all_skus = set(product_data.keys())
    nb_overlap_skus = len(pred_skus.intersection(all_skus))

    return nb_overlap_skus / len(all_skus)


def popularity_bias_at_k(y_preds, x_train, k=3):
    # estimate popularity from training data
    pop_map = collections.defaultdict(lambda : 0)
    num_interactions = 0
    for session in x_train:
        for event in session:
            pop_map[event] += 1
            num_interactions += 1
    # normalize popularity
    pop_map = {k:v/num_interactions for k,v in pop_map.items()}
    all_popularity = []
    for p in y_preds:
        average_pop = sum(pop_map.get(_, 0.0) for _ in p[:k]) / len(p) if len(p) > 0 else 0
        all_popularity.append(average_pop)
    return sum(all_popularity) / len(y_preds)


def precision_at_k(y_preds, y_test, k=3):
    precision_ls = [len(set(_y).intersection(set(_p[:k]))) / len(_p) if _p else 1 for _p, _y in zip(y_preds, y_test)]
    return np.average(precision_ls)


def recall_at_k(y_preds, y_test, k=3):
    recall_ls = [len(set(_y).intersection(set(_p[:k]))) / len(_y) if _y else 1 for _p, _y in zip(y_preds, y_test)]
    return np.average(recall_ls)


def rec_items_distribution_at_k(y_preds, k=3, bin_width=100, debug=True):
    def _calculate_hist(frequencies, bins):
        """ Works out the counts of items in each bucket """
        counts_per_bin = Counter([bisect_right(bins, item) - 1 for item in frequencies])
        counts_per_bin_list = list(counts_per_bin.items())
        empty_bins_indices = [ele for ele in np.arange(len(bins) - 1) if ele not in [
            index for index, _ in counts_per_bin_list
        ]]
        counts_per_bin_list.extend([(index, 0) for index in empty_bins_indices])
        counts_per_bin_sorted = sorted(counts_per_bin_list, key=lambda x: x[0], reverse=False)
        return [y for _, y in counts_per_bin_sorted]

    def _format_results(bins, counts_per_bin):
        """ Formatting results """
        results = {}
        for index, (_, count) in enumerate(zip(bins, counts_per_bin)):
            if bins[index] != bins[index + 1]:
                results[str(bins[index]) + '-' + str(bins[index + 1] - 1)] = count
        return results

    # estimate frequency of recommended items in predictions
    reduce_at_k_preds = [preds[:k] for preds in y_preds]
    counts = Counter(chain.from_iterable(reduce_at_k_preds))

    frequencies = list(counts.values())

    # fixed bin size
    bins = np.arange(np.min(frequencies), np.max(frequencies) + bin_width, bin_width)
    counts_per_bin_sorted = _calculate_hist(frequencies, bins)

    # log bin size
    log_bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    log_bins = np.array([int(round(log_bin, 2)) for log_bin in log_bins])
    counts_per_logbin_sorted = _calculate_hist(frequencies, log_bins)

    test_results = {
        'histogram_fixed': _format_results(bins, counts_per_bin_sorted),
        'histogram_log': _format_results(log_bins, counts_per_logbin_sorted)
    }

    if debug:
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title('Frequency of recommending items across users')

        # debug / visualization
        ax1.bar(bins[:-1], counts_per_bin_sorted,
                width=bin_width, align='edge')

        ax2.bar(log_bins[:-1], counts_per_logbin_sorted,
                width=(log_bins[1:] - log_bins[:-1]), align='edge')
        ax2.set_xscale('log', base=10)

        plt.savefig(os.path.join(current.report_path, 'plots', 'rec_items_distribution_at_k.png'))

    return test_results
