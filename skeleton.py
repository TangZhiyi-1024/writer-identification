import os
import shlex
import argparse
import glob
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
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
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    parser.add_argument('--n_clusters', default=100, type=int,
                        help='number of visual words (for multi-VLAD)')
    parser.add_argument('--max_descs', default=500000, type=int,
                        help='random descriptors for dictionary training (for multi-VLAD)')
    parser.add_argument('--multi_vlad', action='store_true',
                        help='run multi-VLAD + PCA whitening experiment')
    parser.add_argument('--n_codebooks', default=5, type=int,
                        help='number of codebooks for multi-VLAD')
    parser.add_argument('--pca_dim', default=1000, type=int,
                        help='target dimensions after PCA whitening')
    parser.add_argument('--seed', default=42, type=int,
                        help='base random seed for multi-VLAD codebooks')
    return parser

def _build_file_index(folder):
    """
    Build stem->path index by scanning folder recursively.
    """
    index = {}
    for root, _, files in os.walk(folder):
        for fn in files:
            name = fn
            for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ocvmb', '.csv']:
                if name.endswith(p):
                    name = name[:-len(p)]
                    break
            if name not in index:
                index[name] = os.path.join(root, fn)
    return index

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    file_index = _build_file_index(folder)
    all_files = []
    labels = []
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        if not os.path.exists(true_file_name):
            if file_name in file_index:
                true_file_name = file_index[file_name]
            else:
                matches = glob.glob(os.path.join(folder, '**', file_name + '.*'), recursive=True)
                if len(matches) > 0:
                    true_file_name = matches[0]
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def computeDescs(filename):
    """
    Compute SIFT descriptors from image with:
    - keypoint angle fixed to 0
    - Hellinger normalization (l1, then sign-sqrt)
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
            x=kp.pt[0], y=kp.pt[1], size=kp.size, angle=0.0,
            response=kp.response, octave=kp.octave, class_id=kp.class_id
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
    Unified loader:
    - precomputed .pkl.gz descriptors
    - descriptors computed from image file
    """
    if filename.endswith('.pkl.gz'):
        with gzip.open(filename, 'rb') as ff:
            return np.asarray(cPickle.load(ff, encoding='latin1'), dtype=np.float32)
    return computeDescs(filename)

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = min(100, len(files))
    indices = np.random.permutation(len(files))[:max_files]
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    max_descs_per_file = max(1, int(max_descriptors / max(1, len(files))))

    descriptors = []
    missing = 0
    missing_example = None
    for i in tqdm(range(len(files))):
        if not os.path.exists(files[i]):
            missing += 1
            if missing_example is None:
                missing_example = files[i]
            continue
        desc = loadDescriptors(files[i])
        if desc is None or len(desc) == 0:
            continue

        indices = np.random.choice(len(desc),
                                   min(len(desc), int(max_descs_per_file)),
                                   replace=False)
        desc = desc[indices]
        descriptors.append(desc)

    if len(descriptors) == 0:
        if missing > 0:
            print('warning: {} sampled files are missing on disk.'.format(missing))
            print('example missing path: {}'.format(missing_example))
        raise RuntimeError('No descriptors could be loaded from the provided files.')

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters, random_state=None):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=10000,
        random_state=random_state,
        verbose=1
    )
    kmeans.fit(descriptors)
    mus = kmeans.cluster_centers_.astype(np.float32)
    return mus
def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # 转成 float32，OpenCV 要这个类型
    desc = np.asarray(descriptors, dtype=np.float32)
    mus = np.asarray(clusters, dtype=np.float32)

    T, D = desc.shape
    K, _ = mus.shape

    # 最近邻匹配：每个 descriptor 找 1 个最近 cluster
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # knnMatch 返回的是 list，每个元素是长度 k 的 match 列表
    matches = matcher.knnMatch(desc, mus, k=1)

    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )
    for i, m in enumerate(matches):
        best = m[0]  # 只有一个
        k = best.trainIdx  # 最近的 cluster 下标
        assignment[i, k] = 1.0

    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    Dm = mus.shape[1]
    encodings = []

    for f in tqdm(files):
        if not os.path.exists(f):
            encodings.append(np.zeros(K * Dm, dtype=np.float32))
            continue
        desc = loadDescriptors(f)
        if desc is None or len(desc) == 0:
            encodings.append(np.zeros(K * Dm, dtype=np.float32))
            continue
        a = assignments(desc, mus)
        
        T,D = desc.shape
        f_enc = np.zeros( (D*K), dtype=np.float32)
        for k in range(mus.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the 
            # difference to the cluster center than computing the differences
            # first and then select
            idx = np.where(a[:, k] > 0)[0]
            if len(idx) == 0:
                continue

            # residuals: desc[idx] - mus[k]
            diff = desc[idx] - mus[k]  # (#idx, D)

            # sum pooling: 求和得到 1×D
            if gmp:
                y = np.ones(len(idx), dtype=np.float32)
                reg = Ridge(
                    alpha=gamma,
                    fit_intercept=False,
                    solver='sparse_cg',
                    max_iter=500
                )
                reg.fit(diff, y)
                v_k = reg.coef_.astype(np.float32)
            else:
                v_k = diff.sum(axis=0)  # shape: (D,)

            # 把 v_k 放到 f_enc 对应段落
            start = k * D
            end = (k + 1) * D
            f_enc[start:end] = v_k

        # c) power normalization
        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))
        # l2 normalization
        f_enc = f_enc / (np.linalg.norm(f_enc) + 1e-12)
        encodings.append(f_enc)

    encodings = np.stack(encodings, axis=0)
    print("VLAD encodings shape:", encodings.shape)
    return encodings

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """


    # set up labels
    encs_test = np.asarray(encs_test, dtype=np.float32)
    encs_train = np.asarray(encs_train, dtype=np.float32)

    n_test, D = encs_test.shape
    n_neg = encs_train.shape[0]


    # 所有 E-SVM 共享同一套标签：
    # 第 0 个样本 = 正样本(当前 query)，其余 M 个 = 负样本
    y = np.zeros(n_neg + 1, dtype=np.int32)
    y[0] = 1

    def loop(i):
        # 组装当前这个 query 的训练数据：
        # X = [encs_test[i]; encs_train]
        x_pos = encs_test[i][None, :]  # 1 x D
        X = np.vstack([x_pos, encs_train])  # (1 + M) x D

        # 训练 Linear SVM：1 vs all-negatives
        clf = LinearSVC(C=C, class_weight='balanced', max_iter=10000)
        clf.fit(X, y)

        # 取出权重向量 w 作为新的 feature
        w = clf.coef_.reshape(-1)  # (D,)

        # L2 归一化
        w = w / (np.linalg.norm(w) + 1e-12)

        return w[None, :]  # 返回 1 x D，方便后面 concatenate

    new_encs_list = []
    for i in tqdm(range(n_test), desc='E-SVM'):
        new_encs_list.append(loop(i))

    new_encs = np.concatenate(new_encs_list, axis=0)  # N x D
    return new_encs

def l2norm_rows(X):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def build_codebooks(files_train, n_codebooks, n_clusters, max_descs, suffix_tag, seed):
    codebooks = []
    for i in range(n_codebooks):
        local_seed = seed + i
        np.random.seed(local_seed)
        mus_i_name = 'mus_{}_cb{}_k{}_seed{}.pkl.gz'.format(
            suffix_tag, i, n_clusters, local_seed
        )
        if os.path.exists(mus_i_name):
            with gzip.open(mus_i_name, 'rb') as f:
                mus_i = cPickle.load(f)
        else:
            print('> [multi-vlad] load random descriptors for codebook {}'.format(i))
            desc_i = loadRandomDescriptors(files_train, max_descs)
            print('> [multi-vlad] compute dictionary {}'.format(i))
            mus_i = dictionary(desc_i, n_clusters=n_clusters, random_state=local_seed)
            with gzip.open(mus_i_name, 'wb') as fOut:
                cPickle.dump(mus_i, fOut, -1)
        codebooks.append(np.asarray(mus_i, dtype=np.float32))
    return codebooks

def compute_multi_vlad(files, codebooks, powernorm, gmp, gamma):
    all_parts = []
    for i, mus_i in enumerate(codebooks):
        print('> [multi-vlad] encode with codebook {}'.format(i))
        part = vlad(files, mus_i, powernorm, gmp, gamma)
        all_parts.append(part)
    return np.concatenate(all_parts, axis=1).astype(np.float32)


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # TODO
    # mask out distance with itself
    encs = np.asarray(encs, dtype=np.float32)
    # encs 已经是 L2-normalized（VLAD 里做过），所以 dot product 即为 cosine similarity
    sims = np.dot(encs, encs.T)  # T x T
    # cosine distance = 1 - cosine similarity
    dists = 1.0 - sims
    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42) # fix random seed
   
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):
        max_descs = 500000
        print('> load random descriptors')
        descriptors = loadRandomDescriptors(files_train, max_descs)
        print('> loaded {} descriptors:'.format(len(descriptors)))
        # cluster centers
        print('> compute dictionary')
        mus = dictionary(descriptors, n_clusters=100)

        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

  
    # b) VLAD encoding

    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_test = vlad(files_test, mus, args.powernorm, args.gmp, args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)
   
    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname_train = 'enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname_train) or args.overwrite:
        enc_train = vlad(files_train, mus, args.powernorm, args.gmp, args.gamma)
        with gzip.open(fname_train, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname_train, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    enc_test_esvm = esvm(enc_test, enc_train, C=args.C)

    # eval
    print('> evaluate (E-SVM)')
    evaluate(enc_test_esvm, labels_test)

    # g) optional multi-VLAD + PCA whitening (kept separate from a/b/c/d flow)
    if args.multi_vlad:
        suffix_tag = args.suffix.replace('.', '_')
        print('> [multi-vlad] build/load codebooks')
        codebooks = build_codebooks(
            files_train=files_train,
            n_codebooks=args.n_codebooks,
            n_clusters=args.n_clusters,
            max_descs=args.max_descs,
            suffix_tag=suffix_tag,
            seed=args.seed
        )

        mode_tag = 'gmp{}'.format(args.gamma) if args.gmp else 'sum'
        mv_tag = 'mvlad{}_k{}_{}'.format(args.n_codebooks, args.n_clusters, mode_tag)
        mv_train_name = 'enc_train_{}_from_{}.pkl.gz'.format(mv_tag, suffix_tag)
        mv_test_name = 'enc_test_{}_from_{}.pkl.gz'.format(mv_tag, suffix_tag)

        if not os.path.exists(mv_train_name) or args.overwrite:
            print('> [multi-vlad] compute train encodings')
            enc_train_mv = compute_multi_vlad(files_train, codebooks, args.powernorm, args.gmp, args.gamma)
            with gzip.open(mv_train_name, 'wb') as fOut:
                cPickle.dump(enc_train_mv, fOut, -1)
        else:
            with gzip.open(mv_train_name, 'rb') as f:
                enc_train_mv = cPickle.load(f)

        if not os.path.exists(mv_test_name) or args.overwrite:
            print('> [multi-vlad] compute test encodings')
            enc_test_mv = compute_multi_vlad(files_test, codebooks, args.powernorm, args.gmp, args.gamma)
            with gzip.open(mv_test_name, 'wb') as fOut:
                cPickle.dump(enc_test_mv, fOut, -1)
        else:
            with gzip.open(mv_test_name, 'rb') as f:
                enc_test_mv = cPickle.load(f)

        pca_dim = min(args.pca_dim, enc_train_mv.shape[0], enc_train_mv.shape[1])
        print('> [multi-vlad] PCA whitening to {} dims'.format(pca_dim))
        pca = PCA(n_components=pca_dim, whiten=True, random_state=args.seed)
        pca.fit(enc_train_mv)
        enc_train_mv_pca = l2norm_rows(pca.transform(enc_train_mv))
        enc_test_mv_pca = l2norm_rows(pca.transform(enc_test_mv))

        print('> [multi-vlad] evaluate')
        evaluate(enc_test_mv_pca, labels_test)

        print('> [multi-vlad] esvm computation')
        enc_test_mv_esvm = esvm(enc_test_mv_pca, enc_train_mv_pca, C=args.C)
        print('> [multi-vlad] evaluate (E-SVM)')
        evaluate(enc_test_mv_esvm, labels_test)
