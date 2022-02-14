import collections
from reclist.abstractions import RecList, rec_test
from typing import List
import random


class CoveoCartRecList(RecList):

    @rec_test(test_type='stats')
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data
        """
        from reclist.metrics.standard_metrics import statistics
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          self._y_preds)

    @rec_test(test_type='price_homogeneity')
    def price_test(self):
        """
        Measures the absolute log ratio of ground truth and prediction price
        """
        from reclist.metrics.price_homogeneity import price_homogeneity_test
        return price_homogeneity_test(y_test=self.sku_only(self._y_test),
                                      y_preds=self.sku_only(self._y_preds),
                                      product_data=self.product_data,
                                      price_sel_fn=lambda x: float(x['price_bucket'])
                                      if x['price_bucket']
                                      else None
                                      )

    @rec_test(test_type='Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        return coverage_at_k(self.sku_only(self._y_preds),
                             self.product_data,
                             k=10)

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate in which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self.sku_only(self._y_preds),
                             self.sku_only(self._y_test),
                             k=10)

    @rec_test(test_type='hits_distribution')
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across product frequency in training data
        """
        from reclist.metrics.hits import hits_distribution
        return hits_distribution(self.sku_only(self._x_train),
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds),
                                 k=10,
                                 debug=True)

    @rec_test(test_type='distance_to_query')
    def dist_to_query(self):
        """
        Compute the distribution of distance from query to label and query to prediction
        """
        from reclist.metrics.distance_metrics import distance_to_query
        return distance_to_query(self.rec_model,
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds), k=10, bins=25, debug=True)

    def sku_only(self, l: List[List]):
        return [[e['product_sku'] for e in s] for s in l]


class SpotifySessionRecList(RecList):

    @rec_test(test_type='basic_stats')
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data for Next Event Prediction
        """
        from reclist.metrics.standard_metrics import statistics
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          self._y_preds)

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self.uri_only(self._y_preds),
                             self.uri_only(self._y_test),
                             k=10)

    @rec_test(test_type='perturbation_test')
    def perturbation_at_k(self):
        """
        Compute average consistency in model predictions when inputs are perturbed
        """
        from reclist.metrics.perturbation import session_perturbation_test
        from collections import defaultdict
        from functools import partial

        # Step 1: Generate a map from artist uri to track uri
        substitute_mapping = defaultdict(list)
        for track_uri, row in self.product_data.items():
            substitute_mapping[row['artist_uri']].append(track_uri)

        # Step 2: define a custom perturbation function
        def perturb(session, sub_map):
            last_item = session[-1]
            last_item_artist = self.product_data[last_item['track_uri']]['artist_uri']
            substitutes = set(sub_map.get(last_item_artist, [])) - {last_item['track_uri']}
            if substitutes:
                similar_item = random.sample(substitutes, k=1)
                new_session = session[:-1] + [{"track_uri": similar_item[0]}]
                return new_session
            return []

        # Step 3: call test
        return session_perturbation_test(self.rec_model,
                                         self._x_test,
                                         self._y_preds,
                                         partial(perturb, sub_map=substitute_mapping),
                                         self.uri_only,
                                         k=10)

    @rec_test(test_type='shuffle_session')
    def perturbation_shuffle_at_k(self):
        """
        Compute average consistency in model predictions when inputs are re-ordered
        """
        from reclist.metrics.perturbation import session_perturbation_test

        # Step 1: define a custom perturbation function
        def perturb(session):
            return random.sample(session, len(session))

        # Step 2: call test
        return session_perturbation_test(self.rec_model,
                                         self._x_test,
                                         self._y_preds,
                                         perturb,
                                         self.uri_only,
                                         k=10)

    @rec_test(test_type='hits_distribution_by_slice')
    def hits_distribution_by_slice(self):
        """
        Compute the distribution of hit-rate across various slices of data
        """
        from reclist.metrics.hits import hits_distribution_by_slice

        len_map = collections.defaultdict(list)
        for idx, playlist in enumerate(self._x_test):
            len_map[len(playlist)].append(idx)
        slices = collections.defaultdict(list)
        bins = [(x * 5, (x + 1) * 5) for x in range(max(len_map) // 5 + 1)]
        for bin_min, bin_max in bins:
            for i in range(bin_min + 1, bin_max + 1, 1):
                slices[f'({bin_min}, {bin_max}]'].extend(len_map[i])
                del len_map[i]
        assert len(len_map) == 0

        return hits_distribution_by_slice(slices,
                                          self.uri_only(self._y_test),
                                          self.uri_only(self._y_preds),
                                          debug=True)

    @rec_test(test_type='Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        return coverage_at_k(self.uri_only(self._y_preds),
                             self.product_data,
                             # this contains all the track URIs from train and test sets
                             k=10)

    @rec_test(test_type='Popularity@10')
    def popularity_bias_at_k(self):
        """
        Compute average frequency of occurrence across recommended items in training data
        """
        from reclist.metrics.standard_metrics import popularity_bias_at_k
        return popularity_bias_at_k(self.uri_only(self._y_preds),
                                    self.uri_only(self._x_train),
                                    k=10)

    @rec_test(test_type='MRR@10')
    def mrr_at_k(self):
        """
        MRR calculates the mean reciprocal of the rank at which the first
        relevant item was retrieved
        """
        from reclist.metrics.standard_metrics import mrr_at_k
        return mrr_at_k(self.uri_only(self._y_preds),
                        self.uri_only(self._y_test))

    def uri_only(self, playlists: List[dict]):
        return [[track['track_uri'] for track in playlist] for playlist in playlists]


class MovieLensSimilarItemRecList(RecList):
    @rec_test(test_type="stats")
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data
        """
        from reclist.metrics.standard_metrics import statistics
        return statistics(
            self._x_train,
            self._y_train,
            self._x_test,
            self._y_test,
            self._y_preds
        )

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the movie to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(
            self.movie_only(self._y_preds),
            self.movie_only(self._y_test),
            k=10
        )

    @rec_test(test_type='Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible movies which the RS
        recommends based on a set of movies and their respective ratings
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        return coverage_at_k(
            self.movie_only(self._y_preds),
            self.product_data,
            k=10
        )

    @rec_test(test_type='hits_distribution')
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across movie frequency in training data
        """
        from reclist.metrics.hits import hits_distribution
        return hits_distribution(
            self.movie_only(self._x_train),
            self.movie_only(self._x_test),
            self.movie_only(self._y_test),
            self.movie_only(self._y_preds),
            k=10,
            debug=True
        )

    @rec_test(test_type="hits_distribution_by_rating")
    def hits_distribution_by_rating(self):
        """
        Compute the distribution of hit-rate across movie ratings in testing data
        """
        from reclist.metrics.hits import hits_distribution_by_rating
        return hits_distribution_by_rating(
            self._y_test,
            self._y_preds,
            debug=True
        )

    def movie_only(self, movies):
        return [[x["movieId"] for x in y] for y in movies]


class BBCSoundsSimilarItemRecList(RecList):
    @rec_test(test_type="stats")
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data
        """
        from reclist.metrics.standard_metrics import statistics
        return statistics(
            self._x_train,
            self._y_train,
            self._x_test,
            self._y_test,
            self._y_preds
        )

    def resourceid_only(self, resource_ids):
        return [[x["resourceId"] for x in y] for y in resource_ids]

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(
            self.resourceid_only(self._y_preds),
            self.resourceid_only(self._y_test),
            k=10
        )

    @rec_test(test_type='hits_distribution')
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across item frequency in training data
        """
        from reclist.metrics.hits import hits_distribution
        return hits_distribution(
            self.resourceid_only(self._x_train),
            self.resourceid_only(self._x_test),
            self.resourceid_only(self._y_test),
            self.resourceid_only(self._y_preds),
            k=10,
            debug=True
        )

    @rec_test(test_type="hits_distribution_by_agerange")
    def hits_distribution_by_agerange(self):
        """
        Compute the distribution of hit-rate across age ranges in testing data
        """
        from reclist.metrics.hits import hits_distribution_by_agerange
        return hits_distribution_by_agerange(
            self._y_test,
            self._y_preds,
            debug=True
        )

    @rec_test(test_type="hits_distribution_by_gender")
    def hits_distribution_by_gender(self):
        """
        Compute the distribution of hit-rate across gender types in testing data
        """
        from reclist.metrics.hits import hits_distribution_by_gender
        return hits_distribution_by_gender(
            self._y_test,
            self._y_preds,
            debug=True
        )

    @rec_test(test_type='Popularity@10')
    def popularity_bias_at_k(self):
        """
        Compute average frequency of occurrence across recommended items in training data
        """
        from reclist.metrics.standard_metrics import popularity_bias_at_k
        return popularity_bias_at_k(self.resourceid_only(self._y_preds),
                                    self.resourceid_only(self._x_train),
                                    k=10)


class BBCSoundsRecList(RecList):

    def resourceid_only(self, resource_ids):
        return [[x["resourceId"] for x in y] for y in resource_ids]

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(
            self.resourceid_only(self._y_preds),
            self.resourceid_only(self._y_test),
            k=10
        )

    @rec_test(test_type='hits_distribution_custom')
    def hits_distribution_custom(self):
        """
        Compute the distribution of hit-rate across item frequency in training data
        """
        from reclist.metrics.hits import hits_distribution_custom
        return hits_distribution_custom(
            self.resourceid_only(self._x_train),
            self.resourceid_only(self._y_test),
            self.resourceid_only(self._y_preds),
            k=10,
            debug=True
        )

    @rec_test(test_type="hits_distribution_by_agerange")
    def hits_distribution_by_agerange(self):
        """
        Compute the distribution of hit-rate across age ranges in testing data
        """
        from reclist.metrics.hits import hits_distribution_by_agerange
        return hits_distribution_by_agerange(
            self._y_test,
            self._y_preds,
            debug=True
        )

    @rec_test(test_type="hits_distribution_by_gender")
    def hits_distribution_by_gender(self):
        """
        Compute the distribution of hit-rate across gender types in testing data
        """
        from reclist.metrics.hits import hits_distribution_by_gender
        return hits_distribution_by_gender(
            self._y_test,
            self._y_preds,
            debug=True
        )

    @rec_test(test_type='Popularity@10')
    def popularity_bias_at_k(self):
        """
        Compute average frequency of occurrence across recommended items in training data
        """
        from reclist.metrics.standard_metrics import popularity_bias_at_k
        return popularity_bias_at_k(self.resourceid_only(self._y_preds),
                                    self.resourceid_only(self._x_train),
                                    k=10)
    @rec_test(test_type='Entropy@10')
    def shannon_entropy_at_k(self):
        """ Shannon Entropy is a measure of the information content/'randomness' of the prediction set.

        For more ‘random’ data, the Shannon entropy value is higher. For more deterministic signals, it is lower."""
        from reclist.metrics.novelty import shannon_entropy_at_k
        return shannon_entropy_at_k(self.resourceid_only(self._y_preds),
                                    k=10)

    @rec_test(test_type='Entropy_distribution_by_gender@10')
    def shannon_entropy_at_k_gender(self):
        """ Shannon Entropy is a measure of the information content/'randomness' of the prediction set.

        For more ‘random’ data, the Shannon entropy value is higher. For more deterministic signals, it is lower."""
        from reclist.metrics.novelty import shannon_entropy_at_k_user_differential
        return shannon_entropy_at_k_user_differential(self._y_test,
                                                      self.resourceid_only(self._y_preds),
                                                      k=10, user_feature='gender',
                                                      debug=True)

    @rec_test(test_type='GiniIndex@10')
    def gini_index_at_k(self):
        """ The Gini Index is a measure of across k recommendations for n users of
        how equally recommendations are distributed across all items.
          - High values G~1 mean that there is a large inequality across
            items in how often they are recommended
          - G~0 means that all items are recommended roughly the same
            amount across the set of users
        This is a measure of exposure bias resulting from the recommender"""
        from reclist.metrics.novelty import gini_index_at_k
        return gini_index_at_k(self.resourceid_only(self._y_preds),
                               self.product_data,
                               k=10)

    @rec_test(test_type='GiniIndex_distribution_by_age_range@10')
    def gini_index_at_k_age_range(self):
        """ The Gini Index is a measure of across k recommendations for n users of
        how equally recommendations are distributed across all items.
          - High values G~1 mean that there is a large inequality across
            items in how often they are recommended
          - G~0 means that all items are recommended roughly the same
            amount across the set of users
        This is a measure of exposure bias resulting from the recommender"""
        from reclist.metrics.novelty import gini_index_at_k_user_differential
        return gini_index_at_k_user_differential(self.resourceid_only(self._y_preds),
                                                 self._y_test,
                                                 self.product_data,
                                                 k=10,
                                                 debug=True)

    @rec_test(test_type='Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible candidate items which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        return coverage_at_k(self.resourceid_only(self._y_preds),
                             {k: None for k in self.product_data},
                             k=10)

    @rec_test(test_type='Novelty@10')
    def novelty_at_k(self):
        """
        Novelty is the propensity of a recommender to recommend itms that a user has little or no knowledge about.
        Here we use a variation on eq 9. of Silveira et. al. [2017], where popularity is normalised by the number
        of users in the training set. Item popularity is used a proxy for user knowledge. A higher novelty recommender
        recommends less popular items more frequently
        """
        from reclist.metrics.novelty import novelty_at_k
        return novelty_at_k(self.resourceid_only(self._y_preds),
                            self._x_train,
                            k=10)

    @rec_test(test_type='Novelty@10')
    def novelty_at_k_age_range(self):
        """
        Novelty is the propensity of a recommender to recommend itms that a user has little or no knowledge about.
        Here we use a variation on eq 9. of Silveira et. al. [2017], where popularity is normalised by the number
        of users in the training set. Item popularity is used a proxy for user knowledge. A higher novelty recommender
        recommends less popular items more frequently
        """
        from reclist.metrics.novelty import novelty_at_k_user_differential
        return novelty_at_k_user_differential(self._x_train,
                                              self._y_test,
                                              self.resourceid_only(self._y_preds),
                                              k=10, user_feature='age_range',
        debug=True)


    @rec_test(test_type='NDCG@10')
    def ndcg_at_k(self):
        """
        Normalised Discounted Cumulative Gain @ 10
        """
        from reclist.metrics.standard_metrics import ndcg_at_k
        return ndcg_at_k(self.resourceid_only(self._y_preds),
                         self.resourceid_only(self._y_test),
                         k=10)

    @rec_test(test_type='NDCG@10')
    def ndcg_at_k_age_range(self):
        """
        Normalised Discounted Cumulative Gain @ 10
        """
        from reclist.metrics.standard_metrics import ndcg_at_k_user_differential
        return ndcg_at_k_user_differential(self.resourceid_only(self._y_preds),
                                           self.resourceid_only(self._y_test),
                                           self._y_test,
                                           k=10, user_feature='age_range')

    @rec_test(test_type='Personalisation@10')
    def personalisation_at_k(self):
        """
        Aggregated difference measure between user recommendations (unordered).
        Pers@k=1 -> Each recommendation list shares no items with any others (fully disjoint)
        Pers@k=0 -> All recommendation lists are the same for every user
        """
        from reclist.metrics.novelty import personalisation_at_k
        return personalisation_at_k(self.resourceid_only(self._y_preds),
                                    k=10)
