"""
Loading and preparing data for experiments. Some of the datasets are generated using the Matlab
code (see data/matlab_code_data), in order to ensure they are the same training and test points
that were used in the previous paper (Nguyen and Bonilla NIPS (2014)).
The Matlab code to generate data is ``load_data.m``.
"""
import _pickle as cPickle
import pickle
import gzip
import os
import subprocess

import GPy
import numpy as np
import pandas

from savigp.kernel import ExtRBF

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
WISC_DIR = os.path.join(DATA_DIR, 'wisconsin_cancer')
USPS_DIR = os.path.join(DATA_DIR, 'USPS')
MINING_DIR = os.path.join(DATA_DIR, 'mining')
BOSTON_DIR = os.path.join(DATA_DIR, 'boston_housing')
CREEP_DIR = os.path.join(DATA_DIR, 'creep')
SARCOS_DIR = os.path.join(DATA_DIR, 'sarcos')
MNIST_DIR = os.path.join(DATA_DIR, 'mnist')
MNIST8M_DIR = os.path.join(DATA_DIR, 'infimnist')
AIRLINE_DIR = os.path.join(DATA_DIR, 'airline')
SEISMIC_DIR = os.path.join(DATA_DIR, 'seismic')

def normal_generate_samples(n_samples, var, input_dim=3):
    num_samples = n_samples
    num_in = input_dim
    X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, num_in))
    X.sort(axis=0)
    rbf = ExtRBF(num_in, variance=0.5,
                 lengthscale=np.array(np.random.uniform(low=0.1, high=3.0, size=input_dim)),
                 ARD=True)
    white = GPy.kern.White(num_in, variance=var[0, 0])
    kernel = rbf + white
    K = kernel.K(X)
    y = np.empty((num_samples, var.shape[0]))
    for j in range(var.shape[0]):
        y[:, j] = np.reshape(np.random.multivariate_normal(np.zeros(num_samples), K), (num_samples))
    return X, y, rbf


def wisconsin_breast_cancer_data():
    """
    Loads and returns data of Wisconsin breast cancer dataset. Note that ``X`` is standardized.

    Returns
    -------
    data : list
        a list of length = 5, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    Notes
    -----
    Data is directly imported from the Matlab code for AVIGP paper.

    References
    ----------
    * Mangasarian OL, Street WN, Wolberg WH. Breast cancer diagnosis and prognosis via linear
      programming. Oper Res. 1995;43(4);570-7
    """
    data = []
    for i in range(1, 6):
        train = pandas.read_csv(os.path.join(WISC_DIR, 'train_' + str(i) + '.csv'), header=None)
        test = pandas.read_csv(os.path.join(WISC_DIR, 'test_' + str(i) + '.csv'), header=None)
        data.append({
            'train_outputs': train.ix[:, 0].values[:, np.newaxis],
            'train_inputs': train.ix[:, 1:].values,
            'test_outputs': test.ix[:, 0].values[:, np.newaxis],
            'test_inputs': test.ix[:, 1:].values,
            'id': i
        })

    return data


def usps_data():
    """
    Loads and returns data of USPS dataset. Note that ``X`` is standardized. Only digits 4, 7,
    and 9 are included.

    Returns
    -------
    data : list
        A list of length = 5, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    References
    ----------
    * Rasmussen CE, Williams CKI. {G}aussian processes for machine learning. The MIT Press; 2006.
      Data is imported from the Matlab code.
    """
    data = []
    for i in range(1, 6):
        train = pandas.read_csv(os.path.join(USPS_DIR, 'train_' + str(i) + '.csv'), header=None)
        test = pandas.read_csv(os.path.join(USPS_DIR, 'test_' + str(i) + '.csv'), header=None)
        data.append({
            'train_outputs': train.ix[:, 0:2].values,
            'train_inputs': train.ix[:, 3:].values,
            'test_outputs': test.ix[:, 0:2].values,
            'test_inputs': test.ix[:, 3:].values,
            'id': i
        })

    return data


def mining_data():
    """
    Loads and returns data of Coal-mining disasters dataset. See 'get_mine_data.m' to see how data
    is generated.

    Returns
    -------
    data : list
        A list of length = 1, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``. Training and test points are the same.

    References
    ----------
    * Jarrett RG. A note on the intervals between coal-mining disasters. Biometrika.
      1979;66(1):191-3.
    """
    data = []
    train = pandas.read_csv(os.path.join(MINING_DIR, 'data.csv'), header=None)
    data.append({
        'train_outputs': train.ix[:, 0].values[:, np.newaxis],
        'train_inputs': train.ix[:, 1].values[:, np.newaxis],
        'test_outputs': train.ix[:, 0].values[:, np.newaxis],
        'test_inputs': train.ix[:, 1].values[:, np.newaxis],
        'id': 1
    })

    return data


def boston_data():
    """
    Loads and returns data of Boston housing dataset. Note data ``X`` is standardized.

    Returns
    -------
    data : list
        A list of length = 5, where each element is a dictionary which contains
        ``train_outputs``, ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    References
    ----------
    * Harrison Jr D, Rubinfeld DL. Hedonic housing prices and the demand for clean air.
      J Environ Econ Manage. 1978;5(1):81-102.
    """
    data = []
    for i in range(1, 6):
        train = pandas.read_csv(os.path.join(BOSTON_DIR, 'train_' + str(i) + '.csv'), header=None)
        test = pandas.read_csv(os.path.join(BOSTON_DIR, 'test_' + str(i) + '.csv'), header=None)
        data.append({
            'train_outputs': train.ix[:, 0].values[:, np.newaxis],
            'train_inputs': train.ix[:, 1:].values,
            'test_outputs': test.ix[:, 0].values[:, np.newaxis],
            'test_inputs': test.ix[:, 1:].values,
            'id': i
        })

    return data


def abalone_data():
    """
    Loads and returns data of Abalone dataset.

    Returns
    -------
    data : list
        A list of length = 5, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``


    References
    ----------
    * Bache K, Lichman M. {UCI} Machine Learning Repository [Internet]. 2013.
      Available from: http://archive.ics.uci.edu/ml
    """
    data = []
    for i in range(5, 11):
        train = pandas.read_csv(os.path.join(ABALONE_DIR, 'train_' + str(i) + '.csv'), header=None)
        test = pandas.read_csv(os.path.join(ABALONE_DIR, 'test_' + str(i) + '.csv'), header=None)
        data.append({
            'train_outputs': train.ix[:, 0].values[:, np.newaxis],
            'train_inputs': train.ix[:, 1:].values,
            'test_outputs': test.ix[:, 0].values[:, np.newaxis],
            'test_inputs': test.ix[:, 1:].values,
            'id': i
        })

    return data


def creep_data():
    """
    Loads and returns data of Creep dataset.

    Returns
    -------
    data : list
        A list of length = 5, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    References
    ----------
    * Cole D, Martin-Moran C, Sheard AG, Bhadeshia HKDH, MacKay DJC.
    Modelling creep rupture strength of ferritic steel welds. Sci Technol Weld Join. 2000;5(2):81-9.
    """
    data = []
    for i in range(1, 6):
        train = pandas.read_csv(os.path.join(CREEP_DIR, 'train_' + str(i) + '.csv'), header=None)
        test = pandas.read_csv(os.path.join(CREEP_DIR, 'test_' + str(i) + '.csv'), header=None)
        data.append({
            'train_outputs': train.ix[:, 0].values[:, np.newaxis],
            'train_inputs': train.ix[:, 1:].values,
            'test_outputs': test.ix[:, 0].values[:, np.newaxis],
            'test_inputs': test.ix[:, 1:].values,
            'id': i
        })

    return data


def mnist_data():
    """
    Loads and returns data of MNIST dataset for all digits.

    Returns
    -------
    data : list
        A list of length = 1, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    References
    ----------
    * Data is imported from this project: http://deeplearning.net/tutorial/gettingstarted.html
    """
    dataset = os.path.join(MNIST_DIR, 'mnist.pkl.gz')
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)

    if not os.path.isdir(MNIST_DIR): # directory does not exist, download the data
        os.makedirs(MNIST_DIR)
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from {}'.format(origin))
        urllib.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as data_file:
        train_set, valid_set, test_set = pickle.load(data_file, encoding='latin1')

    test_outputs = np.zeros((test_set[1].shape[0], 10))
    test_outputs[np.arange(test_set[1].shape[0]), test_set[1]] = 1
    train_outputs = np.zeros((train_set[1].shape[0], 10))
    train_outputs[np.arange(train_set[1].shape[0]), train_set[1]] = 1
    validation_Y = np.zeros((valid_set[1].shape[0], 10))
    validation_Y[np.arange(valid_set[1].shape[0]), valid_set[1]] = 1

    data = []
    data.append({
        'train_outputs': np.vstack((train_outputs, validation_Y)),
        'train_inputs': np.vstack((train_set[0], valid_set[0])),
        'test_outputs': test_outputs,
        'test_inputs': test_set[0],
        'id': 0
    })

    return data


def mnist8m_data():
    """
    Loads and returns data of MNIST8M dataset for all digits.

    Returns
    -------
    data : list
        A list of length = 1, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    References
    ----------
    * Data is imported from this project: http://deeplearning.net/tutorial/gettingstarted.html
    """
    train_images, train_labels, test_images, test_labels = import_mnist8m()
    data = []
    data.append({
        'train_outputs': train_labels,
        'train_inputs': train_images,
        'test_outputs': test_labels,
        'test_inputs': test_images,
        'id': 0
    })

    return data


def mnist_binary_data():
    """
    Loads and returns data of MNIST dataset for all digits.

    Returns
    -------
    data : list
        A list of length = 1, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    References
    ----------
    * Data is imported from this project: http://deeplearning.net/tutorial/gettingstarted.html
    """
    data = mnist_data()
    # Transform the labels to be -1 on even numbers and 1 on odd.
    to_bin = lambda x: x[1:10:2].sum() - x[0:10:2].sum()
    data[0]['train_outputs'] = np.apply_along_axis(
        to_bin, 1, data[0]['train_outputs']).astype(int)[:, np.newaxis]
    data[0]['test_outputs'] = np.apply_along_axis(
        to_bin, 1, data[0]['test_outputs']).astype(int)[:, np.newaxis]

    return data


def sarcos_data():
    """
    Loads and returns data of SARCOS dataset for joints 4 and 7. Note that ``X`` is standardized.

    Returns
    -------
    data : list
        A list of length = 1, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    References
    ----------
    * Data is originally from this website: http://www.gaussianprocess.org/gpml/data/.
      The data is directly imported from the Matlab code on Gaussian process networks.
      The Matlab code to generate data is 'data/matlab_code_data/sarcos.m'.
    """
    data = []
    train = pandas.read_csv(os.path.join(SARCOS_DIR, 'train_' +'.csv'), header=None)
    test = pandas.read_csv(os.path.join(SARCOS_DIR, 'test_' + '.csv'), header=None)
    data.append({
        'train_outputs': train.ix[:, 0:1].values,
        'train_inputs': train.ix[:, 2:].values,
        'test_outputs': test.ix[:, 0:1].values,
        'test_inputs': test.ix[:, 2:].values,
        'id': 0
    })

    return data


def sarcos_all_joints_data():
    """
    Loads and returns data of SARCOS dataset for all joints.

    Returns
    -------
    data : list
        A list of length = 1, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``

    References
    ----------
    * Data is originally from this website: http://www.gaussianprocess.org/gpml/data/.
    The data here is directly imported from the Matlab code on Gaussian process networks.
    The Matlab code to generate data is 'data/matlab_code_data/sarcos.m'
    """
    data = []
    train = pandas.read_csv(os.path.join(SARCOS_DIR, 'train_all' +'.csv'), header=None)
    test = pandas.read_csv(os.path.join(SARCOS_DIR, 'test_all' + '.csv'), header=None)
    data.append({
        'train_outputs': train.ix[:, 0:6].values,
        'train_inputs': train.ix[:, 7:].values,
        'test_outputs': test.ix[:, 0:6].values,
        'test_inputs': test.ix[:, 7:].values,
        'id': 0
    })

    return data


def airline_data():
    train = pandas.read_csv(os.path.join(AIRLINE_DIR, 'train.csv'), header=None)
    test = pandas.read_csv(os.path.join(AIRLINE_DIR, 'test.csv'), header=None)
    return [{
        'train_outputs': train.values[:, 8:],
        'train_inputs': train.values[:, :8],
        'test_outputs': test.values[:, 8:],
        'test_inputs': test.values[:, :8],
        'id': 0
    }]


def seismic_data():
    data = []
    train = pandas.read_csv(os.path.join(SEISMIC_DIR, 'data.csv'), header=None)
    data.append({
        'train_outputs': train.ix[:, 0:3].values,
        'train_inputs': train.ix[:, 4:5].values,
        'test_outputs': train.ix[:, 0:3].values,
        'test_inputs': train.ix[:, 4:5].values,
        'id': 1
    })
    return data


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D unit8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = int(_read32(bytestream))
    rows = int(_read32(bytestream))
    cols = int(_read32(bytestream))
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def process_mnist(images):
    assert images.shape[3] == 1
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)

    return images


def get_mnist8m_data():
    print("Getting mnist8m data ...")
    os.chdir('experiments/data')
    subprocess.call(["./get_mnist8m_data.sh"])
    os.chdir("../../")
    print("done")


def import_mnist8m():
    if not os.path.isdir(MNIST8M_DIR): # directory does not exist, download the data
        get_mnist8m_data()

    with open(os.path.join(MNIST8M_DIR, "train-patterns.gz")) as f:
        train_images = extract_images(f)
        train_images = process_mnist(train_images)

    with open(os.path.join(MNIST8M_DIR, "train-labels.gz")) as f:
        train_labels = extract_labels(f, one_hot=True)

    with open(os.path.join(MNIST8M_DIR, "test-patterns.gz")) as f:
        test_images = extract_images(f)
        test_images = process_mnist(test_images)

    with open(os.path.join(MNIST8M_DIR, "test-labels.gz")) as f:
        test_labels = extract_labels(f, one_hot=True)

    return train_images, train_labels, test_images, test_labels

