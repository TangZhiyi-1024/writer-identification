import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
import numpy as np
import cv2


def parseArgs(parser):
    parser.add_argument('--labels_test',
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train',
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='.png',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling (placeholder flag)')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP (unused placeholder)')
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')
    parser.add_argument('--n_clusters', default=100, type=int,
                        help='number of visual words')
    parser.add_argument('--max_descs', default=500000, type=int,
                        help='number of random local descriptors for dictionary training')
    return parser


def getFiles(folder, pattern, labelfile):
    """
    returns files and associated labels by reading the labelfile
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels
    """
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()

    all_files = []
    labels = []
    for line in all_lines:
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.jpeg', '.tif', '.ocvmb', '.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p, '')

        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels


def computeDescs(filename):
    """
    Compute SIFT descriptors for an input image.
    Required modifications:
      1) set all keypoint angles to 0
      2) Hellinger normalization of descriptors:
         l1 normalization, then sign-square-root
         (no extra l2 normalization afterwards)
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError('could not load image: {}'.format(filename))

    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    if not keypoints:
        return np.zeros((0, 128), dtype=np.float32)

    keypoints0 = [
        cv2.KeyPoint(
            x=kp.pt[0],
            y=kp.pt[1],
            size=kp.size,
            angle=0.0,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id,
        )
        for kp in keypoints
    ]

    _, desc = sift.compute(img, keypoints0)
    if desc is None or len(desc) == 0:
        return np.zeros((0, 128), dtype=np.float32)

    desc = np.asarray(desc, dtype=np.float32)
    l1 = np.sum(np.abs(desc), axis=1, keepdims=True) + 1e-12
    desc = desc / l1
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    return desc


def loadDescriptors(filename):
    """
    Unified local descriptor loader:
    - if .pkl.gz: load precomputed descriptors
    - else: compute from image with computeDescs
    """
    if filename.endswith('.pkl.gz'):
        with gzip.open(filename, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        return np.asarray(desc, dtype=np.float32)
    return computeDescs(filename)


def loadRandomDescriptors(files, max_descriptors):
    """
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features/images
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    max_files = min(100, len(files))
    indices = np.random.permutation(len(files))[:max_files]
    files = np.array(files)[indices]

    max_descs_per_file = max(1, int(max_descriptors / max(1, len(files))))
    descriptors = []

    for i in tqdm(range(len(files)), desc='sample local descs'):
        if not os.path.exists(files[i]):
            continue
        desc = loadDescriptors(files[i])
        if desc is None or len(desc) == 0:
            continue

        idx = np.random.choice(
            len(desc),
            min(len(desc), max_descs_per_file),
            replace=False
        )
        descriptors.append(desc[idx])

    if len(descriptors) == 0:
        raise RuntimeError('No descriptors could be loaded from the provided files.')

    return np.concatenate(descriptors, axis=0).astype(np.float32)


def dictionary(descriptors, n_clusters):
    """
    return cluster centers for the descriptors
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=10000,
        verbose=1
    )
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_.astype(np.float32)


def assignments(descriptors, clusters):
    """
    compute assignment matrix
    """
    desc = np.asarray(descriptors, dtype=np.float32)
    mus = np.asarray(clusters, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc, mus, k=1)

    assignment = np.zeros((len(descriptors), len(clusters)), dtype=np.float32)
    for i, m in enumerate(matches):
        assignment[i, m[0].trainIdx] = 1.0
    return assignment


def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each file
    """
    K = mus.shape[0]
    D = mus.shape[1]
    encodings = []

    for f in tqdm(files, desc='vlad'):
        if not os.path.exists(f):
            encodings.append(np.zeros(K * D, dtype=np.float32))
            continue

        desc = loadDescriptors(f)
        if desc is None or len(desc) == 0:
            encodings.append(np.zeros(K * D, dtype=np.float32))
            continue

        a = assignments(desc, mus)
        f_enc = np.zeros((D * K), dtype=np.float32)

        for k in range(K):
            idx = np.where(a[:, k] > 0)[0]
            if len(idx) == 0:
                continue
            diff = desc[idx] - mus[k]
            v_k = diff.sum(axis=0)
            start = k * D
            end = (k + 1) * D
            f_enc[start:end] = v_k

        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        # keep VLAD output l2-normalized
        f_enc = f_enc / (np.linalg.norm(f_enc) + 1e-12)
        encodings.append(f_enc)

    encodings = np.stack(encodings, axis=0)
    print('VLAD encodings shape:', encodings.shape)
    return encodings


def esvm(encs_test, encs_train, C=1000):
    """
    compute a new embedding using Exemplar Classification
    """
    encs_test = np.asarray(encs_test, dtype=np.float32)
    encs_train = np.asarray(encs_train, dtype=np.float32)

    n_test, _ = encs_test.shape
    n_neg = encs_train.shape[0]

    y = np.zeros(n_neg + 1, dtype=np.int32)
    y[0] = 1

    new_encs_list = []
    for i in tqdm(range(n_test), desc='E-SVM'):
        x_pos = encs_test[i][None, :]
        X = np.vstack([x_pos, encs_train])
        clf = LinearSVC(C=C, class_weight='balanced', max_iter=10000)
        clf.fit(X, y)
        w = clf.coef_.reshape(-1)
        w = w / (np.linalg.norm(w) + 1e-12)
        new_encs_list.append(w[None, :])

    return np.concatenate(new_encs_list, axis=0)


def distances(encs):
    """
    compute pairwise distances
    """
    encs = np.asarray(encs, dtype=np.float32)
    sims = np.dot(encs, encs.T)
    dists = 1.0 - sims
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists


def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    """
    dist_matrix = distances(encs)
    indices = dist_matrix.argsort()

    n_encs = len(encs)
    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs - 1):
            if labels[indices[r, k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k + 1))
                if k == 0:
                    correct += 1
        mAP.append(np.mean(precisions))
    mAP = np.mean(mAP)
    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42)

    files_train, labels_train = getFiles(args.in_train, args.suffix, args.labels_train)
    print('#train:', len(files_train))

    mus_name = 'mus_sift_from_{}_k{}.pkl.gz'.format(args.suffix.replace('.', '_'), args.n_clusters)
    if not os.path.exists(mus_name) or args.overwrite:
        print('> load random descriptors')
        descriptors = loadRandomDescriptors(files_train, args.max_descs)
        print('> loaded {} descriptors'.format(len(descriptors)))
        print('> compute dictionary')
        mus = dictionary(descriptors, n_clusters=args.n_clusters)
        with gzip.open(mus_name, 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open(mus_name, 'rb') as f:
            mus = cPickle.load(f)

    files_test, labels_test = getFiles(args.in_test, args.suffix, args.labels_test)
    print('#test:', len(files_test))

    enc_test_name = 'enc_test_sift_from_{}.pkl.gz'.format(args.suffix.replace('.', '_'))
    if not os.path.exists(enc_test_name) or args.overwrite:
        print('> compute VLAD for test')
        enc_test = vlad(files_test, mus, args.powernorm, args.gmp, args.gamma)
        with gzip.open(enc_test_name, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(enc_test_name, 'rb') as f:
            enc_test = cPickle.load(f)

    print('> evaluate')
    evaluate(enc_test, labels_test)

    enc_train_name = 'enc_train_sift_from_{}.pkl.gz'.format(args.suffix.replace('.', '_'))
    if not os.path.exists(enc_train_name) or args.overwrite:
        print('> compute VLAD for train (for E-SVM)')
        enc_train = vlad(files_train, mus, args.powernorm, args.gmp, args.gamma)
        with gzip.open(enc_train_name, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(enc_train_name, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    enc_test_esvm = esvm(enc_test, enc_train, C=args.C)

    print('> evaluate (E-SVM)')
    evaluate(enc_test_esvm, labels_test)

