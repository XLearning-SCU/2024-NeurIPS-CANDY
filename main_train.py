import argparse
import copy
import datetime
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import utils
import yaml
from dataset_loader import load_dataset
from engine_train import train_one_epoch
from model import BaseModel, DivideModel, OursModel
from torch.utils import data

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(description="Training")

    # config path
    parser.add_argument("--config_file", type=str, default=None)

    # backbone parameters
    parser.add_argument("--encoder_dim", type=list, nargs="+", default=[])
    parser.add_argument("--embed_dim", type=int, default=0)

    # model parameters
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--start_rectify_epoch", type=int, default=20)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--n_views", type=int, default=2, help="number of views")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes")

    # training setting
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size per GPU"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--warmup_epochs", type=int, default=20, help="epochs to warmup learning rate"
    )
    parser.add_argument(
        "--data_norm",
        type=str,
        default="standard",
        choices=["standard", "min-max", "l2-norm"],
    )
    parser.add_argument("--train_time", type=int, default=5)

    # optimizer parameters
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Initial value of the weight decay. (default: 0)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )

    # data loader and logger
    parser.add_argument(
        "--dataset",
        type=str,
        default="LandUse21",
        choices=[
            "LandUse21",
            "Scene15",
        ],
    )
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument(
        "--data_path", type=str, default="./", help="path to your folder of dataset"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path where to save, empty for no saving",
    )

    parser.add_argument("--print_freq", default=50)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--fp_ratio", type=float, default=0.5)
    parser.add_argument("--use_divide", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")

    parser.add_argument("--singular_thresh", type=float, default=0.2)

    return parser


def train_one_time(args, state_logger):
    utils.fix_random_seeds(args.seed)

    device = torch.device(args.device)

    dataset = load_dataset(args)
    dataset_train, dataset_test = dataset, dataset

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    if args.batch_size > len(sampler_train):
        args.batch_size = len(sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if args.use_divide:
        model = DivideModel(
            n_views=args.n_views,
            layer_dims=args.encoder_dim,
            temperature=args.temperature,
            n_classes=args.n_classes,
            drop_rate=args.drop_rate,
        )
    else:
        model = OursModel(
            n_views=args.n_views,
            layer_dims=args.encoder_dim,
            temperature=args.temperature,
            n_classes=args.n_classes,
            drop_rate=args.drop_rate,
        )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    if args.train_id == 0:
        print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        state_logger.write("Batch size: {}".format(args.batch_size))
        state_logger.write(
            "Start time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        )
        state_logger.write("Train parameters: {}".format(args).replace(", ", ",\n"))
        state_logger.write(model.__repr__())
        state_logger.write(optimizer.__repr__())
        print("Data loaded: there are {:} samples.".format(len(dataset_train)))

    state_logger.write(
        "\n>> Start training {}-th initial, seed: {},".format(args.train_id, args.seed)
    )

    train_state_dict = dict()

    for epoch in range(args.start_epoch, args.epochs):
        args.print_this_epoch = (
            epoch + 1
        ) % args.print_freq == 0 or epoch + 1 == args.epochs
        train_state = train_one_epoch(
            model,
            data_loader_train,
            data_loader_test,
            optimizer,
            device,
            epoch,
            state_logger,
            args,
        )
        if train_state is not None:
            train_state_dict[epoch] = train_state

        if args.save_ckpt:
            torch.save(model, os.path.join(args.output_dir, f"checkpoint_{epoch}"))
        if args.print_this_epoch:
            state_logger.write(
                "Epoch {} K-means: NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}".format(
                    epoch,
                    train_state["nmi"],
                    train_state["ari"],
                    train_state["f"],
                    train_state["acc"],
                )
            )
    return train_state_dict


def main(args):
    start_time = time.time()

    result_avr_def = {"nmi": [], "ari": [], "f": [], "acc": []}

    result_dict = dict()

    batch_scale = args.batch_size / 256
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_scale
        args.min_lr = args.min_blr * batch_scale
    exp_name = args.output_dir + time.strftime("%Y-%m-%d_%H-%M-%S")
    state_logger = utils.FileLogger("log_train.txt")

    for t in range(args.train_time):
        args.train_id = t
        train_state_dict = train_one_time(args, state_logger)
        args.seed = args.seed + 1
        for epoch, train_state in train_state_dict.items():
            result_dict.setdefault(epoch, copy.deepcopy(result_avr_def))
            for k, v in train_state.items():
                result_dict[epoch][k].append(v)

    for epoch, res in result_dict.items():
        for k, v in res.items():
            x = np.asarray(v) * 100

            res[k] = [x.mean(), x.std()]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    for epoch, result_avr in result_dict.items():
        print("Epoch {}:".format(epoch))
        kmeans_r = result_avr
        state_logger.write("title,acc,ari,nmi")
        state_logger.write(
            "concat,{:.2f}({:.2f}),{:.2f}({:.2f}),{:.2f}({:.2f})".format(
                *kmeans_r["acc"], *kmeans_r["ari"], *kmeans_r["nmi"]
            )
        )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            if hasattr(yaml, "FullLoader"):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())

        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)

    EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
    args.data_path = os.path.join(EXPERIMENT_ROOT, "data/multiview")
    folder_name = "_".join(
        [
            args.dataset,
            "msrt",
            str(args.missing_rate),
            "tau",
            str(args.temperature),
            "bs",
            str(args.batch_size),
            "blr",
            str(args.blr),
        ]
    )

    args.output_dir = os.path.join(EXPERIMENT_ROOT, "run/CANDY", folder_name)
    print(f"Output dir: {args.output_dir}")

    args.embed_dim = args.encoder_dim[0][-1]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "visualize")).mkdir(parents=True, exist_ok=True)

    main(args)
