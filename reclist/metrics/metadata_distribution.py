import os

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict, Counter

from reclist.current import current
from reclist.utils.vectorise_sounds_data import generate_genre_dict


def round_up(number):
    return int(number) + (number % 1 > 0)


def genre_distribution_by_gender(enriched_items, y_test, y_preds, k=10, top_genres=10,
                                 first_genre_only=False, debug=True):
    """
    Calculates the distribution of genres by age in testing data
    """

    genre_dict = generate_genre_dict(enriched_items)

    genres_per_gender = defaultdict(list)
    genres_count_per_gender = {}

    for target, pred in zip(y_test, y_preds):
        predicted_genres = []
        target_gender = target[0]["gender"]
        if not target_gender:
            target_gender = 'unknown'
        for resource_obj in pred[:k]:
            try:
                predicted_genres.extend(
                    [genre_dict.get(resource_obj["resourceId"])[0]]) if first_genre_only else predicted_genres.extend(
                    genre_dict.get(resource_obj["resourceId"]))
            except IndexError:
                print(genre_dict.get(resource_obj["resourceId"]))

        genres_per_gender[target_gender].append(Counter(predicted_genres))

    genres_counter = defaultdict(list)
    for gender in sorted(genres_per_gender.keys()):
        gender_len = len(genres_per_gender[gender])
        for result_set in genres_per_gender[gender]:
            for genre, occurrences in result_set.items():
                genres_counter[genre].append(occurrences)
        # padding to account for genres which appear in smaller number of recs
        for genre in genres_counter.keys():
            padded_genre = genres_counter[genre]
            padded_genre += [0] * (gender_len - len(padded_genre))
            genres_counter[genre] = padded_genre
        genres_count_per_gender[gender] = Counter(
            {genre: np.mean(genres_counter[genre]) for genre in genres_counter.keys()}).most_common(top_genres)

    if debug:

        nrows = round_up(len(genres_count_per_gender.keys()) / 2)
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12))
        gen_counter = 0

        for row in ax:
            for col in row:
                try:

                    gender = genres_count_per_gender[list(genres_count_per_gender.keys())[gen_counter]]
                    x_tick_names = [genre[0] for genre in gender]
                    x_tick_idx = list(range(len(x_tick_names)))

                    col.barh(
                        # x_tick_names,
                        x_tick_idx,
                        [genre[1] for genre in gender],
                        align='center', tick_label=x_tick_names
                    )

                    col.set_title(list(genres_count_per_gender.keys())[gen_counter], y=1.0, pad=-14, fontsize=8)
                    col.set_xlabel(f'Mean no. of items \n per genre (top {k} recs)', fontsize=8)

                    gen_counter += 1
                except IndexError:
                    pass
        fig.tight_layout()
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 f'genres_count_per_gender.png'))
        plt.clf()

    return genres_count_per_gender


def genre_distribution_by_agerange(enriched_items, y_test, y_preds, k=10, top_genres=10,
                                   first_genre_only=False, debug=True):
    """
    Calculates the distribution of genre by age range in testing data
    """

    # extract genres
    genre_dict = generate_genre_dict(enriched_items)

    genres_per_age_range = defaultdict(list)
    genres_count_per_age_range = {}

    for target, pred in zip(y_test, y_preds):
        predicted_genres = []
        target_age_range = target[0]["age_range"]
        if not target_age_range:
            target_age_range = 'unknown'
        for resource_obj in pred[:k]:
            try:
                predicted_genres.extend(
                    [genre_dict.get(resource_obj["resourceId"])[0]]) if first_genre_only else predicted_genres.extend(
                    genre_dict.get(resource_obj["resourceId"]))
            except IndexError:
                print(genre_dict.get(resource_obj["resourceId"]))

        genres_per_age_range[target_age_range].append(Counter(predicted_genres))

    genres_counter = defaultdict(list)
    for age_range in sorted(genres_per_age_range.keys()):
        age_range_len = len(genres_per_age_range[age_range])
        for result_set in genres_per_age_range[age_range]:
            for genre, occurrences in result_set.items():
                genres_counter[genre].append(occurrences)
        # padding to account for genres which appear in smaller number of recs
        for genre in genres_counter.keys():
            padded_genre = genres_counter[genre]
            padded_genre += [0] * (age_range_len - len(padded_genre))
            genres_counter[genre] = padded_genre
        genres_count_per_age_range[age_range] = Counter(
            {genre: np.mean(genres_counter[genre]) for genre in genres_counter.keys()}).most_common(top_genres)

    # plots
    if debug:

        nrows = round_up(len(genres_count_per_age_range.keys()) / 2)
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12))
        gen_counter = 0

        for row in ax:
            for col in row:
                try:

                    age_range = genres_count_per_age_range[list(genres_count_per_age_range.keys())[gen_counter]]
                    x_tick_names = [genre[0] for genre in age_range]
                    x_tick_idx = list(range(len(x_tick_names)))

                    col.barh(
                        x_tick_idx,
                        [genre[1] for genre in age_range],
                        align='center', tick_label=x_tick_names
                    )
                    col.set_title(list(genres_count_per_age_range.keys())[gen_counter], y=1.0, pad=-14, fontsize=8)
                    col.set_xlabel(f'Mean no. of items \n per genre (top {k} recs)', fontsize=8)

                    gen_counter += 1
                except IndexError:
                    pass

        fig.tight_layout()
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 f'genres_count_per_age_range.png'))
        plt.clf()

    return genres_count_per_age_range


def masterbrand_distribution_by_gender(enriched_items, y_test, y_preds, k=10, top_masterbrands=10, debug=True):
    """
    Calculates the distribution of genres by age in testing data
    """

    masterbrand_dict = {item['resource_id']: item['master_brand'] for item in enriched_items}

    masterbrand_per_gender = defaultdict(list)
    masterbrand_count_per_gender = {}

    for target, pred in zip(y_test, y_preds):
        predicted_masterbrands = []
        target_gender = target[0]["gender"]
        if not target_gender:
            target_gender = 'unknown'
        for resource_obj in pred[:k]:
            predicted_masterbrands.append(masterbrand_dict.get(resource_obj["resourceId"]))

        masterbrand_per_gender[target_gender].append(Counter(predicted_masterbrands))

    masterbrands_counter = defaultdict(list)
    for gender in sorted(masterbrand_per_gender.keys()):
        gender_len = len(masterbrand_per_gender[gender])
        for result_set in masterbrand_per_gender[gender]:
            for genre, occurrences in result_set.items():
                masterbrands_counter[genre].append(occurrences)
        # padding to account for genres which appear in smaller number of recs
        for genre in masterbrands_counter.keys():
            padded_genre = masterbrands_counter[genre]
            padded_genre += [0] * (gender_len - len(padded_genre))
            masterbrands_counter[genre] = padded_genre
        masterbrand_count_per_gender[gender] = Counter(
            {masterbrand: np.mean(masterbrands_counter[masterbrand]) for masterbrand in
             masterbrands_counter.keys()}).most_common(top_masterbrands)

    if debug:

        nrows = round_up(len(masterbrand_count_per_gender.keys()) / 2)
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        mb_counter = 0

        for row in ax:
            for col in row:
                try:

                    gender = masterbrand_count_per_gender[list(masterbrand_count_per_gender.keys())[mb_counter]]
                    x_tick_names = [masterbrand[0] for masterbrand in gender]
                    x_tick_idx = list(range(len(x_tick_names)))

                    col.barh(
                        # x_tick_names,
                        x_tick_idx,
                        [masterbrand[1] for masterbrand in gender],
                        align='center', tick_label=x_tick_names
                    )

                    col.set_title(list(masterbrand_count_per_gender.keys())[mb_counter], y=1.0, pad=-14, fontsize=8)
                    col.set_xlabel(f'Avg. no. of items per \nmasterbrand (top {k} recs)', fontsize=8)

                    mb_counter += 1
                except IndexError:
                    pass

        fig.tight_layout()
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 f'masterbrand_count_per_gender.png'))
        plt.clf()

    return masterbrand_count_per_gender


def masterbrand_distribution_by_agerange(enriched_items, y_test, y_preds, k=10, top_masterbrands=10, debug=True):
    """
    Calculates the distribution of masterbrand by age range in testing data
    """

    masterbrand_dict = {item['resource_id']: item['master_brand'] for item in enriched_items}

    # hits = defaultdict(int)
    masterbrands_per_age_range = defaultdict(list)
    masterbrand_count_per_age_range = {}

    for target, pred in zip(y_test, y_preds):
        predicted_masterbrands = []
        target_age_range = target[0]["age_range"]
        if not target_age_range:
            target_age_range = 'unknown'
        for resource_obj in pred[:k]:
            predicted_masterbrands.append(masterbrand_dict.get(resource_obj["resourceId"]))

        masterbrands_per_age_range[target_age_range].append(Counter(predicted_masterbrands))

    masterbrands_counter = defaultdict(list)
    for age_range in sorted(masterbrands_per_age_range.keys()):
        age_range_len = len(masterbrands_per_age_range[age_range])
        for result_set in masterbrands_per_age_range[age_range]:
            for masterbrand, occurrences in result_set.items():
                masterbrands_counter[masterbrand].append(occurrences)
        # padding to account for genres which appear in smaller number of recs
        for masterbrand in masterbrands_counter.keys():
            padded_masterbrand = masterbrands_counter[masterbrand]
            padded_masterbrand += [0] * (age_range_len - len(padded_masterbrand))
            masterbrands_counter[masterbrand] = padded_masterbrand
        masterbrand_count_per_age_range[age_range] = Counter(
            {masterbrand: np.mean(masterbrands_counter[masterbrand]) for masterbrand in
             masterbrands_counter.keys()}).most_common(top_masterbrands)

    # plots
    if debug:

        nrows = round_up(len(masterbrand_count_per_age_range.keys()) / 2)
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
        mb_counter = 0

        for row in ax:
            for col in row:
                try:

                    age_range = masterbrand_count_per_age_range[
                        list(masterbrand_count_per_age_range.keys())[mb_counter]]
                    x_tick_names = [masterbrand[0] for masterbrand in age_range]
                    x_tick_idx = list(range(len(x_tick_names)))

                    col.barh(
                        x_tick_idx,
                        [masterbrand[1] for masterbrand in age_range],
                        align='center', tick_label=x_tick_names
                    )
                    col.set_title(list(masterbrand_count_per_age_range.keys())[mb_counter], y=1.0, pad=-14, fontsize=8)
                    col.set_xlabel(f'Avg. no. of items per \nmasterbrand (top {k} recs)', fontsize=8)

                    mb_counter += 1
                except IndexError:
                    pass

        fig.tight_layout()
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 f'masterbrands_count_per_age_range.png'))
        plt.clf()

    return masterbrand_count_per_age_range
