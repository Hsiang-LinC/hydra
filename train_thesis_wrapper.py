import argparse
import logging
import os
import sys
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Ensure repo root is on sys.path for thesis imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.neural_networks.models import DatasetName, ResNet18
from thesis.neural_networks.utils import get_tinyimagenet_loaders

from models.layers import SubnetConv, SubnetLinear
from trainer import adv as trainer_adv
from trainer import base as trainer_base
from utils import eval as eval_utils
from utils.schedules import get_lr_policy, get_optimizer
from utils.logging import create_subdirs, save_checkpoint
from utils.model import (
    prepare_model,
    show_gradients,
    initialize_scaled_score,
    scale_rand_init,
    sanity_check_paramter_updates,
)

TRAINERS = {
    "base": trainer_base.train,
    "adv": trainer_adv.train,
}
VAL_METHODS = {
    "base": eval_utils.base,
    "adv": eval_utils.adv,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train thesis ResNet18 on TinyImageNet with hydra trainers."
    )
    # Run config
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--result-dir", type=str, default="./output_hydra")
    parser.add_argument(
        "--exp-mode",
        choices=("prune", "finetune"),
        default="prune",
        help="Hydra exp_mode to run (pretrain not supported in this wrapper).",
    )
    parser.add_argument("--trainer", choices=("base", "adv"), default="base")
    parser.add_argument("--val-method", choices=("base", "adv"), default=None)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--source-net", type=str, default="", help="Path to checkpoint.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")

    # Optimizer / LR schedule
    parser.add_argument("--optimizer", choices=("sgd", "adam", "rmsprop"), default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine", "step"), default="cosine")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0.0001)

    # Data loader
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--dataloader-normalize",
        action="store_true",
        default=False,
        help="Enable dataset normalization (not recommended with NormalizedModel).",
    )

    # TRADES / adversarial training
    parser.add_argument("--epsilon", type=float, default=0.031)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.0078)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=1.0)
    parser.add_argument("--distance", choices=("l_inf", "l_2"), default="l_inf")
    parser.add_argument("--const-init", action="store_true", default=False)

    # Pruning / finetuning compatibility with hydra args
    parser.add_argument(
        "--scores-init-type",
        choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
        default=None,
    )
    parser.add_argument("--scaled-score-init", action="store_true", default=False)
    parser.add_argument("--scale-rand-init", action="store_true", default=False)
    parser.add_argument("--snip-init", action="store_true", default=False)
    parser.add_argument("--freeze-bn", action="store_true", default=False)
    parser.add_argument("--save-dense", action="store_true", default=False)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument(
        "--prune-type",
        choices=("unstructured", "structured", "nm"),
        default="unstructured",
        help="Pruning granularity: unstructured, structured (dim=0), or nm (N:M).",
    )
    parser.add_argument("--structured-dim", type=int, default=0)
    parser.add_argument("--nm-n", type=int, default=0)
    parser.add_argument("--nm-m", type=int, default=0)
    parser.add_argument("--exclude-outer", action="store_true", default=True)
    parser.add_argument(
        "--include-outer",
        action="store_false",
        dest="exclude_outer",
        help="Allow pruning first/last layer (compat with hydra args).",
    )
    parser.add_argument(
        "--dense",
        action="store_false",
        dest="use_subnet",
        help="Use dense layers instead of subnet layers (no score-based pruning).",
    )
    parser.set_defaults(use_subnet=True)

    return parser.parse_args()


def _make_eval_loader(loader, batch_size, num_workers):
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _has_popup_scores(model):
    for module in model.modules():
        if hasattr(module, "popup_scores"):
            return True
    return False


def _load_checkpoint(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _replace_with_subnet(parent, name, module):
    if isinstance(module, nn.Conv2d):
        new_module = SubnetConv(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
        )
        new_module.weight.data.copy_(module.weight.data)
        if module.bias is not None and new_module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        setattr(parent, name, new_module)
        return
    if isinstance(module, nn.Linear):
        new_module = SubnetLinear(
            module.in_features, module.out_features, bias=module.bias is not None
        )
        new_module.weight.data.copy_(module.weight.data)
        if module.bias is not None and new_module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        setattr(parent, name, new_module)
        return

    for child_name, child in module.named_children():
        _replace_with_subnet(module, child_name, child)


def main():
    args = parse_args()
    if not args.source_net:
        raise ValueError("--source-net is required for prune/finetune runs.")
    if args.val_method is None:
        args.val_method = "adv" if args.trainer == "adv" else "base"

    if args.dataloader_normalize:
        logging.warning(
            "dataloader normalization is enabled; ResNet18 already normalizes inputs."
        )

    tiny_root = os.path.expanduser("~/datasets/tiny-imagenet-200")
    if not os.path.isdir(tiny_root):
        raise FileNotFoundError(
            f"TinyImageNet not found at {tiny_root} (required by thesis loader)."
        )

    # Setup output dirs
    result_main_dir = os.path.join(args.result_dir, args.exp_mode)
    os.makedirs(result_main_dir, exist_ok=True)
    n = len(next(os.walk(result_main_dir))[-2]) if os.path.exists(result_main_dir) else 0
    result_sub_dir = os.path.join(
        result_main_dir,
        f"{n + 1}--k-{args.k:.2f}_{args.trainer}_lr-{args.lr}_epochs-"
        f"{args.epochs}_warmuplr-{args.warmup_lr}_warmupepochs-{args.warmup_epochs}",
    )
    create_subdirs(result_sub_dir)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a"))
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    # Data loaders (TinyImageNet loader path is fixed in thesis utils)
    dataloaders = get_tinyimagenet_loaders(
        batch_size=args.batch_size,
        normalize=args.dataloader_normalize,
        validation_fraction=args.validation_fraction,
        shuffle=True,
        num_workers=args.num_workers,
        seed_generator=args.seed,
    )
    train_loader = dataloaders.train
    test_loader = _make_eval_loader(
        dataloaders.test, args.test_batch_size, args.num_workers
    )

    # Model
    model = ResNet18(DatasetName.TinyImageNet)
    if args.use_subnet:
        _replace_with_subnet(model, "model", model.model)
    if len(gpu_list) > 1 and use_cuda:
        model = nn.DataParallel(model, gpu_list)
    model = model.to(device)
    logger.info(model)

    if args.exp_mode == "prune" and not _has_popup_scores(model):
        raise ValueError("--exp-mode prune requires subnet layers (popup_scores).")

    # Apply hydra pruning/finetuning setup.
    prepare_model(model, args)

    # Optimizer / LR schedule
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    trainer = TRAINERS[args.trainer]
    val = VAL_METHODS[args.val_method]

    writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))

    # Load source checkpoint (required for prune/finetune).
    state_dict = _load_checkpoint(args.source_net, device)
    model.load_state_dict(state_dict, strict=False)

    if args.scaled_score_init:
        initialize_scaled_score(model)
    if args.scale_rand_init:
        scale_rand_init(model, args.k)
    if args.snip_init:
        raise ValueError("snip-init requires pretrain; not supported in this wrapper.")

    if args.evaluate or args.exp_mode in ["prune", "finetune"]:
        prec1, _ = val(model, device, test_loader, criterion, args, writer)
        logger.info(f"Validation accuracy {args.val_method} for source-net: {prec1}")
        if args.evaluate:
            return

    show_gradients(model)

    last_ckpt = copy.deepcopy(model.state_dict())
    best_prec1 = 0.0
    total_epochs = args.epochs + args.warmup_epochs
    for epoch in range(total_epochs):
        lr_policy(epoch)
        trainer(
            model,
            device,
            train_loader,
            None,
            criterion,
            optimizer,
            epoch,
            args,
            writer,
        )

        prec1, _ = val(model, device, test_loader, criterion, args, writer, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "thesis_resnet18",
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
            save_dense=args.save_dense,
            model=model,
        )
        logger.info(
            f"Epoch {epoch}, val-method {args.val_method}, validation accuracy {prec1}, "
            f"best_prec {best_prec1}"
        )
        if _has_popup_scores(model):
            sw, ss = sanity_check_paramter_updates(model, last_ckpt)
            logger.info(
                f"Sanity check (exp-mode: {args.exp_mode}): Weight update - {sw}, "
                f"Scores update - {ss}"
            )


if __name__ == "__main__":
    main()
