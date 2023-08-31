import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
import yaml
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from ema_pytorch import EMA

from model.models import get_models_class
from utils import Config, init_seeds, gather_tensor, DataLoaderDDP, print0


def get_model(opt, load_epoch):
    DIFFUSION, NETWORK = get_models_class(opt.model_type, opt.net_type)
    diff = DIFFUSION(nn_model=NETWORK(**opt.network),
                     **opt.diffusion,
                     device=device,
                     )
    diff.to(device)
    target = os.path.join(opt.save_dir, "ckpts", f"model_{load_epoch}.pth")
    print0("loading model at", target)
    checkpoint = torch.load(target, map_location=device)
    ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
    ema.to(device)
    ema.load_state_dict(checkpoint['EMA'])
    model = ema.ema_model
    model.eval()
    return model


class ClassifierDict(nn.Module):
    def __init__(self, feat_func, time_list, name_list, base_lr, epoch, img_shape, local_rank, num_classes=10):
        super(ClassifierDict, self).__init__()
        self.feat_func = feat_func
        self.times = time_list
        self.names = name_list
        self.classifiers = nn.ModuleDict()
        self.optims = {}
        self.schedulers = {}
        self.loss_fn = nn.CrossEntropyLoss()

        for time in self.times:
            feats = self.feat_func(torch.zeros(1, *img_shape).to(device), time)
            if self.names is None:
                self.names = list(feats.keys()) # all available names

            for name in self.names:
                key = self.make_key(time, name)
                layers = nn.Linear(feats[name].shape[1], num_classes)
                layers = torch.nn.parallel.DistributedDataParallel(
                    layers.to(device), device_ids=[local_rank], output_device=local_rank)
                optimizer = torch.optim.Adam(layers.parameters(), lr=base_lr)
                scheduler = CosineAnnealingLR(optimizer, epoch)
                self.classifiers[key] = layers
                self.optims[key] = optimizer
                self.schedulers[key] = scheduler

    def train(self, x, y):
        self.classifiers.train()
        for time in self.times:
            feats = self.feat_func(x, time)
            for name in self.names:
                key = self.make_key(time, name)
                representation = feats[name].detach()
                logit = self.classifiers[key](representation)
                loss = self.loss_fn(logit, y)

                self.optims[key].zero_grad()
                loss.backward()
                self.optims[key].step()
    
    def test(self, x):
        outputs = {}
        with torch.no_grad():
            self.classifiers.eval()
            for time in self.times:
                feats = self.feat_func(x, time)
                for name in self.names:
                    key = self.make_key(time, name)
                    representation = feats[name].detach()
                    logit = self.classifiers[key](representation)
                    pred = logit.argmax(dim=-1)
                    outputs[key] = pred
        return outputs

    def make_key(self, t, n):
        return str(t) + '/' + n

    def get_lr(self):
        key = self.make_key(self.times[0], self.names[0])
        optim = self.optims[key]
        return optim.param_groups[0]['lr']

    def schedule_step(self):
        for time in self.times:
            for name in self.names:
                key = self.make_key(time, name)
                self.schedulers[key].step()


def train(opt):
    def test():
        preds = {k: [] for k in classifiers.optims.keys()}
        accs = {}
        labels = []
        for image, label in tqdm(valid_loader, disable=(local_rank!=0)):
            outputs = classifiers.test(image.to(device))
            for key in outputs:
                preds[key].append(outputs[key])
            labels.append(label.to(device))

        for key in preds:
            preds[key] = torch.cat(preds[key])
        label = torch.cat(labels)
        dist.barrier()
        label = gather_tensor(label)
        for key in preds:
            pred = gather_tensor(preds[key])
            accs[key] = (pred == label).sum().item() / len(label)
        return accs

    yaml_path = opt.config
    ep = opt.epoch
    use_amp = opt.use_amp
    grid_search = opt.grid
    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    if ep == -1:
        ep = opt.n_epoch - 1
    model = get_model(opt, ep)

    epoch = opt.linear['n_epoch']
    batch_size = opt.linear['batch_size']
    base_lr = opt.linear['lrate']

    if grid_search:
        time_list = [1, 11, 21] if opt.model_type == 'DDPM' else [3, 4, 5]
        name_list = None
    else:
        time_list = [opt.linear['timestep']]
        name_list = [opt.linear['blockname']]

    train_set = CIFAR10("./data", train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ]))
    valid_set = CIFAR10("./data", train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    train_loader, sampler = DataLoaderDDP(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader, _ = DataLoaderDDP(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
    )

    feat_func = partial(model.get_feature, norm=False, use_amp=use_amp)
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (base_lr, DDP_multiplier))
    base_lr *= DDP_multiplier
    classifiers = ClassifierDict(feat_func, time_list, name_list,
                                 base_lr, epoch, opt.network['image_shape'], local_rank).to(model.device)

    for e in range(epoch):
        sampler.set_epoch(e)
        pbar = tqdm(train_loader, disable=(local_rank!=0))
        for i, (image, label) in enumerate(pbar):
            pbar.set_description("[epoch %d / iter %d]: lr: %.1e" % (e, i, classifiers.get_lr()))
            classifiers.train(image.to(device), label.to(device))
        classifiers.schedule_step()

    accs = test()
    for key in accs:
        print0("[key %s]: Test acc: %.2f" % (key, accs[key] * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument("--grid", action='store_true', default=False)
    opt = parser.parse_args()
    print0(opt)

    local_rank = opt.local_rank
    init_seeds(no=local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank

    train(opt)
