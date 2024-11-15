import math
import os
import sys

import numpy as np
import torch
import utils
from model import BaseModel
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from utils import MetricLogger, SmoothedValue, adjust_learning_config


def train_one_epoch(
    model: BaseModel,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    state_logger=None,
    args=None,
):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 50
    if args.print_this_epoch:
        data_loader = enumerate(
            metric_logger.log_every(data_loader_train, print_freq, header)
        )
    else:
        data_loader = enumerate(data_loader_train)

    model.train(True)
    optimizer.zero_grad()

    for data_iter_step, (idx, samples, label1, label2, id1, id2) in data_loader:
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        lr = adjust_learning_config(optimizer, smooth_epoch, args)
        mmt = args.momentum

        for i in range(args.n_views):
            samples[i] = samples[i].to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=False):
            loss = model(
                samples, mmt, epoch < args.start_rectify_epoch, args.singular_thresh
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.print_this_epoch:
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    if args.print_this_epoch:
        print("Averaged stats:", metric_logger)
        eval_result = evaluate(model, data_loader_test, device, epoch, args)
    else:
        eval_result = None
    return eval_result


def evaluate(
    model: BaseModel,
    data_loader_test: DataLoader,
    device: torch.device,
    epoch: int,
    args=None,
):
    model.eval()
    with torch.no_grad():
        features_all = torch.zeros(args.n_views, args.n_sample, args.embed_dim).to(
            device
        )
        labels_all = torch.zeros(args.n_sample, dtype=torch.long).to(device)
        for indexs, samples, labels, label2, id1, id2 in data_loader_test:
            for i in range(args.n_views):
                samples[i] = samples[i].to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            features = model.extract_feature(samples)

            for i in range(args.n_views):
                features_all[i][indexs] = features[i]

            labels_all[indexs] = labels

        features_cat = features_all.permute(1, 0, 2).reshape(args.n_sample, -1)
        features_cat = torch.nn.functional.normalize(features_cat, dim=-1).cpu().numpy()
        kmeans_label = KMeans(n_clusters=args.n_classes, random_state=0).fit_predict(
            features_cat
        )

    nmi, ari, f, acc = utils.evaluate(np.asarray(labels_all.cpu()), kmeans_label)
    result = {"nmi": nmi, "ari": ari, "f": f, "acc": acc}
    return result
