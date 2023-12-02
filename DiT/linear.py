import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast

from diffusion import create_diffusion
from download import find_model
from models import DiT_XL_2
import sys
sys.path.append("..") 
from utils import init_seeds, gather_tensor, DataLoaderDDP, print0


class LatentCodeDataset(Dataset):
    # warning: needs A LOT OF memory to load these datasets !
    def __init__(self, dataset, train=True, num_copies=10):
        if train:
            code_path = [f"latent_codes/{dataset}/train_code_{i}.npy" for i in range(num_copies)]
            label_path = f"latent_codes/{dataset}/train_label.npy"
        else:
            code_path = [f"latent_codes/{dataset}/test_code_0.npy"]
            label_path = f"latent_codes/{dataset}/test_label.npy"

        self.code = []
        for p in code_path:
            with open(p, 'rb') as f:
                data = np.load(f)
                self.code.append(data)
        with open(label_path, 'rb') as f:
            self.label = np.load(f)

        print0(f"Code shape: {len(self.code)} x {self.code[0].shape}")
        print0("Label shape:", self.label.shape)

    def __getitem__(self, index):
        replica = random.randrange(len(self.code))
        code = self.code[replica][index]
        label = self.label[index]
        return code, label

    def __len__(self):
        return len(self.code[0])


def get_model(device):
    model = DiT_XL_2().to(device)
    state_dict = find_model(f"DiT-XL-2-256x256.pt")
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(None) # 1000-len betas
    return model, diffusion


def denoise_feature(code, model, timestep, blockname, use_amp):
    '''
        Args:
            `image`: Latent codes. (-1, 4, 32, 32) tensor.
            `timestep`: Time step to extract features. int.
            `blockname`: Block to extract features. str.
        Returns:
            Collected feature map.
    '''
    x = code.to(device)
    t = torch.tensor([timestep]).to(device).repeat(x.shape[0])
    noise = torch.randn_like(x)
    x_t = diffusion.q_sample(x, t, noise=noise)
    y_null = torch.tensor([1000] * x.shape[0], device=device)

    with torch.no_grad():
        with autocast(enabled=use_amp):
            _, acts = model(x_t, t, y_null, ret_activation=True)
        feat = acts[blockname].float().detach()
        # (-1, 256, 1152)
        # we average pool across the sequence dimension to extract
        # a 1152-dimensional vector of features per example
        return feat.mean(dim=1)


class Classifier(nn.Module):
    def __init__(self, feat_func, base_lr, epoch, num_classes):
        super(Classifier, self).__init__()
        self.feat_func = feat_func
        self.loss_fn = nn.CrossEntropyLoss()

        hidden_size = feat_func(next(iter(valid_loader))[0]).shape[-1]
        layers = nn.Sequential(
            # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, num_classes),
        )
        layers = torch.nn.parallel.DistributedDataParallel(
            layers.to(device), device_ids=[local_rank], output_device=local_rank)
        self.classifier = layers
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=base_lr)
        self.scheduler = CosineAnnealingLR(self.optim, epoch)

    def train(self, x, y):
        self.classifier.train()
        feat = self.feat_func(x)
        logit = self.classifier(feat)
        loss = self.loss_fn(logit, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def test(self, x):
        with torch.no_grad():
            self.classifier.eval()
            feat = self.feat_func(x)
            logit = self.classifier(feat)
            pred = logit.argmax(dim=-1)
            return pred

    def get_lr(self):
        return self.optim.param_groups[0]['lr']

    def schedule_step(self):
        self.scheduler.step()


def train(model, timestep, blockname, epoch, base_lr, use_amp):
    def test():
        preds = []
        labels = []
        for image, label in tqdm(valid_loader, disable=(local_rank!=0)):
            pred = classifier.test(image.to(device))
            preds.append(pred)
            labels.append(label.to(device))

        pred = torch.cat(preds)
        label = torch.cat(labels)
        dist.barrier()
        pred = gather_tensor(pred)
        label = gather_tensor(label)
        acc = (pred == label).sum().item() / len(label)
        return acc

    print0(f"Feature extraction: time = {timestep}, name = {blockname}")
    feat_func = partial(denoise_feature, model=model, timestep=timestep, blockname=blockname, use_amp=use_amp)
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (base_lr, DDP_multiplier))
    base_lr *= DDP_multiplier
    num_classes = 10 if opt.dataset == 'cifar' else 200

    classifier = Classifier(feat_func, base_lr, epoch, num_classes).to(device)

    for e in range(epoch):
        sampler.set_epoch(e)
        pbar = tqdm(train_loader, disable=(local_rank!=0))
        for i, (image, label) in enumerate(pbar):
            pbar.set_description("[epoch %d / iter %d]: lr: %.1e" % (e, i, classifier.get_lr()))
            classifier.train(image.to(device), label.to(device))
        classifier.schedule_step()

    acc = test()
    print0("Test acc: %.2f" % (acc * 100))


def get_default_time(dataset, t):
    if t > 0:
        return t
    else:
        return {'cifar': 121, 'tiny': 81}[dataset]


def get_default_name(dataset, b):
    if b != 'layer-0':
        return b
    else:
        return {'cifar': 'layer-13', 'tiny': 'layer-13'}[dataset]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cifar', type=str, choices=['cifar', 'tiny'])
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--time', type=int, default=0)
    parser.add_argument('--name', type=str, default='layer-0')
    opt = parser.parse_args()

    local_rank = opt.local_rank
    init_seeds(no=local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank
    model, diffusion = get_model(device)

    train_set = LatentCodeDataset(opt.dataset, train=True)
    valid_set = LatentCodeDataset(opt.dataset, train=False)
    train_loader, sampler = DataLoaderDDP(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    valid_loader, _ = DataLoaderDDP(
        valid_set,
        batch_size=opt.batch_size,
        shuffle=False,
    )

    # default timestep & blockname values
    opt.time = get_default_time(opt.dataset, opt.time)
    opt.name = get_default_name(opt.dataset, opt.name)

    print0(opt)
    train(model, timestep=opt.time, blockname=opt.name, epoch=opt.epoch, base_lr=opt.lr, use_amp=opt.use_amp)
