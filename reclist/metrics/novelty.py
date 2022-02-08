""" Collection of novelty-based metrics """
from collections import defaultdict
import numpy as np


def gini_index_at_k(y_preds, candidate_list, k: int=10, debug: bool=False) -> float:
    no_items = len(set(candidate_list))
    all_preds = []
    for _p in y_preds:
        all_preds += _p[:k]
    all_preds_flattened = np.asarray(all_preds)
    items, counts = np.unique(all_preds_flattened, return_counts=True)
    counts.sort()
    num_recommended_items = counts.shape[0]
    total_num = len(y_preds)*k
    idx = np.arange(no_items-num_recommended_items+1, no_items+1)
    gini_index = np.sum((2*idx-no_items-1)*counts)/total_num
    gini_index /= no_items
    return gini_index

def gini_index_at_k_user_differential(y_preds, y_test, candidate_list, k=10,
                                      debug=False, user_feature='age_range', **kwargs):
    breakdown = _breakdown_preds_by_user_feature(y_test, y_preds,
                                                 user_feature=user_feature)

    return _apply_func_to_breakdown(gini_index_at_k, breakdown, candidate_list,
                                    k=k, debug=debug)
    return {key: gini_index_at_k(val, candidate_list, k=k, debug=debug, **kwargs)
            for key, val in sorted(breakdown.items(), key=lambda x:x[0])}


def shannon_entropy_at_k(y_preds, k: int=10, debug: bool=False) -> float:
    all_preds = []
    for _p in y_preds:
        all_preds += _p[:k]
    all_preds_flattened = np.asarray(all_preds)
    items, counts = np.unique(all_preds_flattened, return_counts=True)
    total_num = len(y_preds)*k
    p = counts/total_num
    return (-p*np.log(p)).sum()


def shannon_entropy_at_k_user_differential(y_test, y_preds, k=10,
                                           debug=False, user_feature='gender', **kwargs):
    breakdown = _breakdown_preds_by_user_feature(y_test, y_preds,
                                                 user_feature=user_feature)

    return _apply_func_to_breakdown(shannon_entropy_at_k, breakdown,
                                    k=k, debug=debug)


def novelty_at_k(y_preds, x_train, k=10, debug=False):
    all_inters = []
    for user in x_train:
        for interaction in user:
            all_inters.append(interaction['resourceId'])
    all_inters = np.asarray(all_inters)
    items, counts = np.unique(all_inters, return_counts=True)
    pop_lookup = {i[0]: i[1] for i in zip(items, counts)}
    msi = []
    n = 0
    no_users = len(x_train)  # each entry is a user
    for _p in y_preds:
        self_information = 0
        n += 1
        for i in _p[:k]:
            try:
                self_information += np.sum(-np.log2(pop_lookup[i]/no_users))
            except KeyError:
                self_information += np.sum(-np.log2(1/no_users))
        msi.append(self_information/k)
    return sum(msi)/n


def novelty_at_k_user_differential(x_train, y_test, y_preds, k=10,
                                   debug=False, user_feature='gender', **kwargs):
    breakdown = _breakdown_preds_by_user_feature(y_test, y_preds,
                                                 user_feature=user_feature)
    return _apply_func_to_breakdown(novelty_at_k, breakdown, x_train, k=k, debug=debug)


def personalisation_at_k(y_preds, k=10, debug=False):
    import itertools
    import tqdm
    _id = 0
    item_ids = {}
    for i in y_preds:
        for j in i:
            if j not in item_ids:
                item_ids[j] = _id
                _id += 1
    total_similarity = 0

    # iterate over each pair of users and compute similarity
    no_users = len(y_preds)
    print(no_users)
    no_combinations = no_users*(no_users-1)/2
    user_vector_cache = {}

    def get_vector(cache, id, k, item_ids, y_preds):
        try:
            return cache[id]
        except KeyError:
            user_vector_ids = set([item_ids[i] for i in y_preds[id][:k]])
            cache[id] = user_vector_ids
            return user_vector_ids

    # Make this speedy by using a.b/|a|/|b|, and caching a,b throughout
    # a, b are len(no_items), all zeros apart from indices of the top k items, which are ==1
    # therefore a.b = number of items in both a, b
    # As only k recommendations, |a| == |b|, |a|^2 = k, which factorises out
    # so compute (a & b) for all unique a,b pairs
    # divide by k
    # divide by number of pairs
    # n.b. doing it the full way (numpy vectors) for 10k users took 2hrs, this takes 30s
    for id1, id2 in tqdm.tqdm(itertools.combinations(range(no_users), 2),
                              total=no_combinations):
        v1 = get_vector(user_vector_cache, id1, k, item_ids, y_preds)
        v2 = get_vector(user_vector_cache, id2, k, item_ids, y_preds)
        total_similarity += len(v1 & v2)
    return 1-(total_similarity/no_combinations/k)


def _breakdown_preds_by_user_feature(y_test, y_preds, user_feature='gender'):
    breakdown = defaultdict(list)
    for _t, _p in zip(y_test, y_preds):
        target_user_feature = _t[0][user_feature]
        if not target_user_feature:
            target_user_feature = 'unknown'
        breakdown[target_user_feature].append(_p)
    return breakdown


def _apply_func_to_breakdown(func, breakdown, *args, **kwargs):
    return {key: func(val, *args, **kwargs)
            for key, val in sorted(breakdown.items(),
                                   key=lambda x:x[0])}
