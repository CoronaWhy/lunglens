import os
import argparse
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from time import sleep

import numpy as np
import torch
import torchvision
from torch.nn.parallel import DataParallel
from torch.nn import Conv2d
from torch.nn.init import kaiming_normal_
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import albumentations as A

from simclr.model import load_optimizer, save_model
from simclr.modules import SimCLR, NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model
from simclr.utils import yaml_config_hook

from lunglens.core import *
from lunglens.loaders import *

def train(args, data_loader, model, criterion, optimizer, writer):
    loss_epoch = 0

    # super batch contain N slices per M scans
    for step, (superbatch0, superbatch1) in enumerate(tqdm(data_loader)):
        num_scans, slices_per_scan = superbatch0.shape[:2]

        # TODO: check what does non_blocking param do and if we need it

        # squash to list batches to one batch
        # [N, M, ...] -> [1, N * M, ...]
        x_i = squash_scan_batches(superbatch0).to(args.device)
        x_j = squash_scan_batches(superbatch1).to(args.device)

        # compute encodings for joined batches
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        if args.single_scan_loss:
            # split joined batch into single batch per scan
            # so we calculate loss looking at a single scan at a time
            z_i = split_scans(z_i, num_scans, slices_per_scan)
            z_j = split_scans(z_j, num_scans, slices_per_scan)
        
        loss = 0
        optimizer.zero_grad()

        # enumerate batches in superbatches
        #print(z_i.shape, z_j.shape)
        if args.single_scan_loss:
            for (z_i_batch, z_j_batch) in zip(z_i, z_j):
                loss += criterion(z_i_batch, z_j_batch)
            do_step = True
        else:
            # not single scan loss
            if z_i.shape[0] == criterion.batch_size:
                loss += criterion(z_i, z_j)
                do_step = True
            else:
                do_step = False
                # probably end of epoch, not enough data

        if do_step:
            loss.backward()
            optimizer.step()

            if args.nr == 0 and step > 0 and step % 20 == 0:
                print(f"Step [{step}/{len(data_loader)}]\t Loss: {loss.item()}")

            if args.nr == 0:
                writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
                args.global_step += 1

            loss_epoch += loss.item()
    return loss_epoch


class NT_Xent_caclulator(NT_Xent):

    def forward(self, z_i, z_j, visualize=False):

        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss, sim * self.temperature


def validate(data_loader, model, visualize=False):

    for i, (superbatch0, superbatch1) in enumerate(data_loader):
        x_i = squash_scan_batches(superbatch0).to(args.device)
        x_j = squash_scan_batches(superbatch1).to(args.device)
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        z_i, z_j = z_i.cpu().detach().numpy(), z_j.cpu().detach().numpy()
        if i == 0:
            z_i_vec, z_j_vec = np.copy(z_i), np.copy(z_j)
        else:
            z_i_vec = np.concatenate((z_i_vec, z_i))
            z_j_vec = np.concatenate((z_j_vec, z_j))
        #print(i, x_i.shape, x_j.shape, z_i.shape, z_j.shape, z_i_vec.shape, z_j_vec.shape)

    num_pairs = z_i_vec.shape[0]
    loss_calculator = NT_Xent_caclulator(batch_size=num_pairs, \
                                         temperature=0.5, device='cpu', world_size=1)
    loss, sim = loss_calculator(torch.from_numpy(z_i_vec), torch.from_numpy(z_j_vec))
    #print('validation loss =', loss)
    if visualize:
        ur = sim[:num_pairs, num_pairs:]
        bl = sim[num_pairs:, :num_pairs]
        positives = np.concatenate((np.diag(ur), np.diag(bl))).flatten()
        allpairs = np.concatenate((ur, bl)).flatten()
        bins = np.linspace(start=-1, stop=1, num=500)
        positives = np.histogram(positives, bins)[0]
        assert positives.sum() == num_pairs * 2
        negatives = np.histogram(allpairs, bins)[0] - positives
        assert negatives.sum() == 2*num_pairs * (num_pairs - 1)
        bins = (bins[:-1] + bins[1:]) / 2
        fig, ax = plt.subplots(2, 2)
        ax[0, 1].imshow(ur, cmap="gist_gray", interpolation='none')
        ax[1, 0].imshow(bl, cmap='gist_gray', interpolation='none')
        ax[0, 0].plot(bins, negatives / negatives.sum(), label='negs')
        ax[0, 0].plot(bins, positives / positives.sum(), label='pos')
        ax[0, 0].grid(True)
        ax[0, 0].legend()
        plt.show()

    return loss.numpy()



def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tfms = A.Compose([
        A.Resize(512, 512),
        A.RandomCrop(384, 384),
        ToColorTensor()
    ])
    if torch.cuda.device_count() == 1:
        if torch.cuda.get_device_name() == 'GeForce MX130':
            tfms = A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(192, 192)
            ])

    # dataset = RandomSlicerDataset(
    #     args.datasets_root, img_tfm(tfms),
    #     args.slices_per_scan, args.inter_slice_distance
    # )
    dataset = PatchedDataset(args.datasets_root, target_size=(128, 128))

    loader = DataLoader(dataset, batch_size=args.scans_per_batch, shuffle=True)

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    #
    # override input layer to make it monochrome
    encoder.conv1 = Conv2d(1, encoder.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    kaiming_normal_(encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
    #
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
    criterion = NT_Xent(
        args.slices_per_scan if args.single_scan_loss else args.scans_per_batch * args.slices_per_scan,
        args.temperature, args.device, args.world_size
    )

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
    dataset.set_tst_mode()
    validation_loss_trc = [validate(loader, model, visualize=True)]
    dataset.set_trn_mode()
    print('validation loss before train:', validation_loss_trc[0])
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        t_start = datetime.now()
        loss_epoch = train(args, loader, model, criterion, optimizer, writer)
        t_end = datetime.now()
        train_time = (t_end - t_start)

        dataset.set_tst_mode()
        validation_loss_trc.append(validate(loader, model, epoch == args.epochs-1))
        dataset.set_trn_mode()

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(loader)}\t lr: {round(lr, 5)}"
            )
            print('validation loss:', validation_loss_trc[-1], '\t\ttrain_time:', train_time)
            sleep(0.2)
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)
    #
    plt.plot(validation_loss_trc)
    plt.grid(True)
    plt.title('validation loss trace')
    plt.show()
    #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./simclr/config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.model_path = './saved_models'
    args.workers = 0
    args.datasets_root = './data/prepared'
    args.datasets_root = './data/patches'
    args.log_dir = './logs'
    args.epochs = 30

    args.scans_per_batch = 4
    args.scans_per_batch = 8
    args.slices_per_scan = 4
    args.inter_slice_distance = 50
    args.single_scan_loss = False

    print(f'Batch size: {args.scans_per_batch * args.slices_per_scan}')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    main(0, args)

    print('done')
