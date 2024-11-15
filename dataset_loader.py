import os.path

import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp
import torch
from numpy.random import randint
from scipy import sparse


def load_mat(args):
    data_X = []
    label_y = None

    if args.dataset == "Scene15":
        mat = sio.loadmat(os.path.join(args.data_path, "Scene_15.mat"))
        X = mat["X"][0]
        data_X.append(X[0].astype("float32"))
        data_X.append(X[1].astype("float32"))
        label_y = np.squeeze(mat["Y"])

    elif args.dataset == "LandUse21":
        mat = sio.loadmat(os.path.join(args.data_path, "LandUse_21.mat"))
        data_X.append(sparse.csr_matrix(mat["X"][0, 1]).A)
        data_X.append(sparse.csr_matrix(mat["X"][0, 2]).A)

        label_y = np.squeeze(mat["Y"]).astype("int")

    elif args.dataset == "Reuters":
        mat = sio.loadmat(os.path.join(args.data_path, "Reuters_dim10.mat"))
        data_X = []  # 18758 samples
        data_X.append(np.vstack((mat["x_train"][0], mat["x_test"][0])))
        data_X.append(np.vstack((mat["x_train"][1], mat["x_test"][1])))
        label_y = np.squeeze(np.hstack((mat["y_train"], mat["y_test"])))

    elif args.dataset == "Caltech101":
        mat = sio.loadmat(
            os.path.join(args.data_path, "2view-caltech101-8677sample.mat")
        )
        X = mat["X"][0]
        data_X.append(X[0].T)
        data_X.append(X[1].T)
        print(X[0].shape, X[1].shape)
        label_y = np.squeeze(mat["gt"]) - 1

    elif args.dataset == "NUSWIDE":
        mat = sio.loadmat(
            os.path.join(args.data_path, "nuswide_deep_2_view.mat")
        )
        data_X.append(mat["Img"])
        data_X.append(mat["Txt"])
        label_y = np.squeeze(mat["label"].T)

    else:
        raise KeyError(f"Unknown Dataset {args.dataset}")

    if args.data_norm == "standard":
        for i in range(args.n_views):
            data_X[i] = skp.scale(data_X[i])
    elif args.data_norm == "l2-norm":
        for i in range(args.n_views):
            data_X[i] = skp.normalize(data_X[i])
    elif args.data_norm == "min-max":
        for i in range(args.n_views):
            data_X[i] = skp.minmax_scale(data_X[i])

    # Control the randomness of the data
    rng = np.random.RandomState(1234)

    label_y_view2 = label_y.copy()
    id1 = np.arange(data_X[0].shape[0])
    id2 = id1.copy()
    if args.fp_ratio > 0:
        for i in range(1, args.n_views):
            inx = np.arange(data_X[i].shape[0])
            rng.shuffle(inx)
            inx = inx[0 : int(args.fp_ratio * data_X[i].shape[0])]
            _inx = np.array(inx)
            rng.shuffle(_inx)
            data_X[i][inx] = data_X[i][_inx]
            label_y_view2[inx] = label_y_view2[_inx]
            id2[inx] = id1[_inx]

    args.n_sample = data_X[0].shape[0]
    return data_X, label_y, label_y_view2, id1, id2


def load_dataset(args):
    data, label1, label2, id1, id2 = load_mat(args)
    dataset = MultiviewDataset(args.n_views, data, label1, label2, id1, id2)
    return dataset


class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label1, label2, id1, id2):
        super(MultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.label1 = label1 - np.min(label1)
        self.label2 = label2 - np.min(label2)
        self.id1 = id1
        self.id2 = id2

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype("float32")))
        label1 = torch.tensor(self.label1[idx], dtype=torch.long)
        label2 = torch.tensor(self.label2[idx], dtype=torch.long)
        id1 = torch.tensor(self.id1[idx], dtype=torch.long)
        id2 = torch.tensor(self.id2[idx], dtype=torch.long)
        return idx, data, label1, label2, id1, id2
