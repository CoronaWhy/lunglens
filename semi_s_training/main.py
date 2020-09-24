import os
import numpy as np
import torch
import torchvision
import argparse
from datetime import datetime

from torch.nn.parallel import DataParallel

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from simclr.model import load_optimizer, save_model
from simclr.modules import SimCLR, NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model
from simclr.utils import yaml_config_hook


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(args, encoder, n_features)
    if args.reload or args.start_epoch:
        epoch_n = args.start_epoch if args.start_epoch else args.epoch_num
        model_fp = os.path.join(args.model_path, f'checkpoint_{epoch_n}.tar')
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        print(f'Loaded from epoch #{epoch_n}')
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.device, args.world_size)

    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        run_log = os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(run_log)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./simclr/config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.model_path = './semi_s_training/simclr/save'
    args.workers = 0
    args.dataset_dir = './simclr/datasets'
    args.log_dir = './semi_s_training/simclr/logs'
    args.epochs = 1
    args.batch_size = 32

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    main(0, args)

    print('done')
