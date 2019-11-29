#DICOS data loader is based on 20 newsgroup loader, created by Olivier Grisel
"""
ORIGINAL DESCRIPTION OF 20 NEWSGROUP DATA LOADER
Caching loader for the 20 newsgroups text classification dataset


The description of the dataset is available on the official website at:

    http://people.csail.mit.edu/jrennie/20Newsgroups/

Quoting the introduction:

    The 20 Newsgroups data set is a collection of approximately 20,000
    newsgroup documents, partitioned (nearly) evenly across 20 different
    newsgroups. To the best of my knowledge, it was originally collected
    by Ken Lang, probably for his Newsweeder: Learning to filter netnews
    paper, though he does not explicitly mention this collection. The 20
    newsgroups collection has become a popular data set for experiments
    in text applications of machine learning techniques, such as text
    classification and text clustering.

This dataset loader will download the recommended "by date" variant of the
dataset and which features a point in time split between the train and
test sets. The compressed dataset size is around 14 Mb compressed. Once
uncompressed the train set is 52 MB and the test set is 34 MB.
"""
# Copyright (c) 2011 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

import os
from os.path import dirname, join
import logging
import tarfile
import pickle
import shutil
import re
import codecs
import sklearn

import numpy as np
import scipy.sparse as sp

from sklearn.datasets import get_data_home
from sklearn.datasets import load_files
from sklearn.datasets.base import _pkl_filepath
from sklearn.datasets.base import _fetch_remote
from sklearn.datasets.base import RemoteFileMetadata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils import deprecated
from sklearn.utils import _joblib
from sklearn.utils import check_random_state, Bunch

logger = logging.getLogger(__name__)


ARCHIVE = 'datasets.tar.gz' #loading the dataset archive

CACHE_NAME = "datasets.pkz"
TRAIN_FOLDER = "datasets/ocr-train"
TEST_FOLDER = "datasets/ocr-test"



def download_20newsgroups(target_dir, cache_path):
    return _download_20newsgroups(target_dir, cache_path)


def _download_20newsgroups(target_dir, cache_path):
    """Download the 20 newsgroups data and stored it as a zipped pickle."""
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    module_path = dirname(__file__)
    archive_path=join(module_path, ARCHIVE)
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)

    # Store a zipped pickle
    cache = dict(train=load_files(train_path, encoding='latin1'),
                 test=load_files(test_path, encoding='latin1'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)
    return cache


def fetch_ocr(data_home=None, subset='train', categories=None,
                       shuffle=True, random_state=42,
                       remove=(),
                       download_if_missing=True):
    """Load the filenames and data from the OCR dataset \
(classification).

    Download it if necessary.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========

    Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------
    data_home : optional, default: None
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    categories : None or collection of string or unicode
        If None (default), load all the categories.
        If not None, list of category names to load (other categories
        ignored).

    shuffle : bool, optional
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    remove : tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

        'headers' follows an exact standard; the other filters are not always
        correct.

    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    bunch : Bunch object with the following attribute:
        - bunch.data: list, length [n_samples]
        - bunch.target: array, shape [n_samples]
        - bunch.filenames: list, length [n_samples]
        - bunch.DESCR: a description of the dataset.
        - bunch.target_names: a list of categories of the returned data,
          length [n_classes]. This depends on the `categories` parameter.
    """

    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    twenty_home = os.path.join(data_home, "20news_home")
    cache = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)

    if cache is None:
        if download_if_missing:
            logger.info("Downloading OCR dataset. "
                        "This may take a few minutes.")
            cache = _download_20newsgroups(target_dir=twenty_home,
                                           cache_path=cache_path)
        else:
            raise IOError('OCR dataset not found')

    if subset in ('train', 'test'):
        data = cache[subset]
    elif subset == 'all':
        data_lst = list()
        target = list()
        filenames = list()
        for subset in ('train', 'test'):
            data = cache[subset]
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)

        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)


    if categories is not None:
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        # Sort the categories to have the ordering of the labels
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.in1d(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        # searchsorted to have continuous labels
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    return data


def fetch_ocr_vectorized(subset="train",
                                  remove=(),
                                  data_home=None,
                                  download_if_missing=True, return_X_y=False):
    """Load the OCR dataset and vectorize it into token counts \
(classification).

    Download it if necessary.

    This is a convenience function; the transformation is done using the
    default settings for
    :class:`sklearn.feature_extraction.text.CountVectorizer`. For more
    advanced usage (stopword filtering, n-gram extraction, etc.), combine
    fetch_20newsgroups with a custom
    :class:`sklearn.feature_extraction.text.CountVectorizer`,
    :class:`sklearn.feature_extraction.text.HashingVectorizer`,
    :class:`sklearn.feature_extraction.text.TfidfTransformer` or
    :class:`sklearn.feature_extraction.text.TfidfVectorizer`.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality          130107
    Features                  real
    =================   ==========

    ORIGINAL PIECE: Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------
    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    remove : tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

    data_home : optional, default: None
        Specify an download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : boolean, default=False.
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    Returns
    -------
    bunch : Bunch object with the following attribute:
        - bunch.data: sparse matrix, shape [n_samples, n_features]
        - bunch.target: array, shape [n_samples]
        - bunch.target_names: a list of categories of the returned data,
          length [n_classes].
        - bunch.DESCR: a description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20
    """
    data_home = get_data_home(data_home=data_home)
    filebase = '20newsgroup_vectorized'
    if remove:
        filebase += 'remove-' + ('-'.join(remove))
    target_file = _pkl_filepath(data_home, filebase + ".pkl")

    # we shuffle but use a fixed seed for the memoization
    data_train = fetch_ocr(data_home=data_home,
                                    subset='train',
                                    categories=None,
                                    shuffle=True,
                                    random_state=12,
                                    #remove=remove,
                                    download_if_missing=download_if_missing)

    data_test = fetch_ocr(data_home=data_home,
                                   subset='test',
                                   categories=None,
                                   shuffle=True,
                                   random_state=12,
                                   #remove=remove,
                                   download_if_missing=download_if_missing)

    if os.path.exists(target_file):
        X_train, X_test = _joblib.load(target_file)
    else:
        vectorizer = CountVectorizer(dtype=np.int16)
        X_train = vectorizer.fit_transform(data_train.data).tocsr()
        X_test = vectorizer.transform(data_test.data).tocsr()
        _joblib.dump((X_train, X_test), target_file, compress=9)

    # the data is stored as int16 for compactness
    # but normalize needs floats
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    normalize(X_train, copy=False)
    normalize(X_test, copy=False)

    target_names = data_train.target_names

    if subset == "train":
        data = X_train
        target = data_train.target
    elif subset == "test":
        data = X_test
        target = data_test.target
    elif subset == "all":
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))
    else:
        raise ValueError("%r is not a valid subset: should be one of "
                         "['train', 'test', 'all']" % subset)


    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 target_names=target_names)
