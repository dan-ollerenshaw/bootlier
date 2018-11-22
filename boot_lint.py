__author__ = 'Daniel Ollerenshaw'
__version__ = '0.0.0'

# Imports
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.kde import gaussian_kde

try:
    import peakutils
except ModuleNotFoundError:
    raise Exception('peakutils is not installed. Please install it first.')


class EmptyDataException(Exception):
    """ To flag cases where dataset is empty either pre or post trimming.
    """
    pass


#%%
class Bootlier():
    """ General class for conducting a "bootlier" test for identifying
        statistical outliers in a given dataset.

        This is the "shortcut" version of the test (add info in README)

        The main advantage of the bootlier test is that it does not depend on
        the distribution of the population.

        Main sources:
        http://sankhya.isical.ac.in/search/65_3/2003030.pdf
        https://www.bundesbank.de/Redaktion/EN/Downloads/Publications/Discussion_Paper_1/2013/2013_02_01_dkp_02.pdf?__blob=publicationFile
        https://github.com/jodeleeuw/Bootlier
        https://github.com/mtpatter/kaiba

        Params:
            data :          1-D numpy array of floats.
            n_bootstraps :  int, optional
                            number of times to sample from the data
            trim_type :     str, optional
                            one of ('percentile', 'scalar')
                            percentile indicates that your trim should take off
                            a percentage of the data, scalar indicates that
                            your trim should take off a fixed number of
                            datapoints.
            trim :          int, optional
                            If tail_type is 'both', this will apply the value
                            of "trim" to each tail.
            tail_type :     str, optional
                            one of ('left_only','right_only','both')
                            whether to trim from both tails or just one tail
            sample_frac:    float, optional
                            proportion of data to sample each bootstrap.
                            for large samples, it is recommended to set this
                            somewhere less than 1, as the probability of not
                            sampling a single outlier decreases with sample size.

        Example usage:
            my_data = np.random.normal(size=100)
            # add an outlier:
            my_data = np.append(my_data, 7)

            boot = Bootlier(my_data)

            # inspect mtm distribution:
            sns.distplot(boot.mtm)

            # test for outliers:
            outliers = boot.find_outliers()
    """
    def __init__(self,
                 data,
                 n_bootstraps=1000,
                 trim_type='scalar',
                 trim=3,
                 tail_type='both',
                 sample_frac=1):
        self.data = data
        self.n_bootstraps = n_bootstraps
        self.trim_type = trim_type
        self.trim = trim
        self.tail_type = tail_type
        self.sample_frac = sample_frac

        # sense-checks
        assert trim_type in ('percentile', 'scalar'), 'Error, invalid trim_type.'
        assert tail_type in ('left_only', 'right_only', 'both'), 'Error, invalid tail_type.'
        if not any(self.data):
            raise EmptyDataException

        # as part of __init__, generate the mtm distribution
        self.mtm = self._gen_mean_diffs(data=self.data,
                                        n=self.n_bootstraps,
                                        trim_type=self.trim_type,
                                        trim=self.trim,
                                        tail_type=self.tail_type,
                                        sample_frac=self.sample_frac)


    def _gen_mean_diffs(self, data, n, trim_type, trim, tail_type, sample_frac):
        """ This function bootstraps a dataset n times, returning an array of
            "mean - trimmed-mean" statistics.

            The way this is currently set up creates a large matrix. This
            speeds up the processing but can cause a memory error if the
            dataset is huge.
            #TODO: chunk large datasets

            Using the trim param:
            If trim_type is "scalar":
              "trim" values will be removed from the tail(s) depending on
              the tail_type param.
            If trim_type is "percentile":
              "trim" percent of values will be removed the tail(s) depending on
              the tail_type param.
        """
        data_len = len(data)
        size = (n, round(data_len * sample_frac)) # matrix dimensions
        samples = np.random.choice(data, size=size, replace=True)
        samples.sort(axis=1) # sort to identify tails by position
        sample_means = np.mean(samples, axis=1)

        if trim_type == 'percentile':
            trim = max(round((trim/100) * data_len * sample_frac), 1) # minimum of 1
        if tail_type == 'left_only':
            clipped = np.delete(samples, np.s_[:trim], axis=1)
        elif tail_type == 'right_only':
            clipped = np.delete(samples, np.s_[-trim:], axis=1)
        elif tail_type == 'both':
            left_clip = np.delete(samples, np.s_[:trim], axis=1)
            clipped = np.delete(left_clip, np.s_[-trim:], axis=1)

        if not clipped.size:
            raise EmptyDataException

        clipped_means = np.mean(clipped, axis=1)

        return sample_means - clipped_means


    def _is_multimodal(self, vector, hyper_sensitive=False):
        """ This function decides whether a distribution is unimodal or
            multimodal.

            thres params: if 0, it will pick up ANY peak regardless of the
            size. If 1, it will pick up zero peaks.

            I think thres = 0 is too sensitive...
        """
        thres = 0 if hyper_sensitive else 0.01 # still pretty sensitive
        peaks = peakutils.indexes(vector, thres=thres)
        if len(peaks) > 1:
            return True
        else:
            return False


    def _evaluate_kde(self, kde):
        """ Evaluate a given KDE.

            There are two choices here: (1) how many points to evaluate it for,
            and (2) the min and max values for the evaluation.

            (1) is hardcoded to 500 for now.

            For (2), for the shortcut version I've chosen to take min and max
            from the mtm distribution itself.
        """
        x = np.linspace(min(self.mtm), max(self.mtm), 500)
        return kde.evaluate(x)


    def has_outliers(self):
        """ Simply test whether the mtm distribution is uni- or multimodal.

            Recommend having a large number of bootstraps for this. (10,000+)
        """
        kde = gaussian_kde(self.mtm, bw_method='silverman') # take default bandwidth
        kde_evaluated = self._evaluate_kde(kde)
        return self._is_multimodal(kde_evaluated)


    def find_outliers(self, show_steps=False):
        """ Quick and dirty adaptation of Bootlier_full method.

            I've set the return type to namedtuple, rather than list,
            to distinguish between LHS outliers and RHS outliers.
        """
        Outliers = namedtuple('outliers', ['LHS', 'RHS'])
        outliers = Outliers([], []) # initialise
        sorted_data = sorted(self.data)
        if self.has_outliers():
            print('Data contains outliers.')
        else:
            print('Data contains no outliers.')
            return outliers

        def remove_from_rhs(data):
            return data[:-1], sorted_data[-1]

        def remove_from_lhs(data):
            return data[1:], sorted_data[0]

        if self.tail_type == 'both':
            # in this case, we remove furthest from median
            while True:
                median = np.median(sorted_data)
                lh_diff, rh_diff = median - sorted_data[0], sorted_data[-1] - median
                removal_func, tail = (remove_from_lhs, 'LHS')\
                                    if lh_diff >= rh_diff\
                                    else (remove_from_rhs, 'RHS')
                sorted_data, outlier = removal_func(sorted_data)
                getattr(outliers, tail).append(outlier)
                if show_steps:
                    print('{} removed from data.'.format(outlier))
                internal_boot = Bootlier(sorted_data,
                                         n_bootstraps=self.n_bootstraps,
                                         trim_type=self.trim_type,
                                         trim=self.trim,
                                         tail_type=self.tail_type,
                                         sample_frac=self.sample_frac)
                if internal_boot.has_outliers():
                    if show_steps:
                        print('Data still contains outliers, re-testing.')
                        print('New mtm distribution:')
                        sns.distplot(internal_boot.mtm)
                        plt.show()
                        plt.close()
                else:
                    if show_steps:
                        print('Data no longer has outliers!')
                        print('New mtm distribution:')
                        sns.distplot(internal_boot.mtm)
                        plt.show()
                        plt.close()
                    return outliers
        else:
            # remove from specified tail only.
            removal_func, tail = (remove_from_lhs, 'LHS')\
                                if self.tail_type == 'left_only'\
                                else (remove_from_rhs, 'RHS')
            while True:
                sorted_data, outlier = removal_func(sorted_data)
                getattr(outliers, tail).append(outlier)
                if show_steps:
                    print('{} removed from data.'.format(outlier))
                internal_boot = Bootlier(sorted_data,
                                         n_bootstraps=self.n_bootstraps,
                                         trim_type=self.trim_type,
                                         trim=self.trim,
                                         tail_type=self.tail_type,
                                         sample_frac=self.sample_frac)
                if internal_boot.has_outliers():
                    if show_steps:
                        print('Data still contains outliers, re-testing.')
                        print('New mtm distribution:')
                        sns.distplot(internal_boot.mtm)
                        plt.show()
                        plt.close()
                else:
                    if show_steps:
                        print('Data no longer has outliers!')
                        print('New mtm distribution:')
                        sns.distplot(internal_boot.mtm)
                        plt.show()
                        plt.close()
                    return outliers


if __name__ == '__main__':
    print("Import me, don't run me!")
