import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
import yaml
import torch.nn as nn
from datasets import get_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from ema_pytorch import EMA

from model.models import get_models_class
from model.block import TimeEmbedding
from utils import Config, init_seeds, reduce_tensor, gather_tensor, DataLoaderDDP, print0


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

''' Train a two-layer noise-conditional MLP classifier.
    This training script is similar to `linear.py` which performs linear probing test.
'''

class Classifier(nn.Module):
    def __init__(self, feat_func, blockname, dim, num_classes):
        super(Classifier, self).__init__()
        self.feat_func = feat_func
        self.blockname = blockname
        self.time_emb = TimeEmbedding(dim, augment_dim=0)
        self.cls = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, num_classes)
        )

    def forward(self, x, t):
        with torch.no_grad():
            x = self.feat_func(x.to(device), t=t)
            x = x[self.blockname].detach()
        return self.cls(x + self.time_emb(t, aug_label=None))


class DDPM:
    def __init__(self, device, n_T=1000, steps=20):
        self.device = device
        self.n_T = n_T
        self.test_timesteps = (torch.arange(0, self.n_T, self.n_T // steps) + 1).long().tolist()

    def train(self, x):
        _t = torch.randint(1, self.n_T + 1, (x.shape[0], ))
        return x, _t.to(self.device)

    def test(self, x, t):
        _t = torch.full((x.shape[0], ), t)
        return x, _t.to(self.device)


class EDM:
    def __init__(self, device, steps=18):
        self.device = device
        self.steps = steps
        self.test_timesteps = range(1, steps + 1)

    def train(self, x):
        _t = torch.randint(1, self.steps + 1, (x.shape[0], ))
        return x, _t.to(self.device)

    def test(self, x, t):
        _t = torch.full((x.shape[0], ), t)
        return x, _t.to(self.device)


def train(opt):
    def test(t):
        preds = []
        labels = []
        for image, label in tqdm(valid_loader, disable=(local_rank!=0)):
            with torch.no_grad():
                model.eval()
                logit = model(*diff.test(image, t))
                pred = logit.argmax(dim=-1)
                preds.append(pred)
                labels.append(label.to(device))

        pred = torch.cat(preds)
        label = torch.cat(labels)
        dist.barrier()
        pred = gather_tensor(pred)
        label = gather_tensor(label)
        acc = (pred == label).sum().item() / len(label)
        return acc

    yaml_path = opt.config
    ep = opt.epoch
    use_amp = opt.use_amp
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
    blockname = opt.linear['blockname']

    mode = opt.model_type
    if mode == 'DDPM':
        diff = DDPM(device)
    elif mode == 'EDM':
        diff = EDM(device)
    else:
        raise NotImplementedError

    train_set = get_dataset(name=opt.dataset, root="./data", train=True, flip=True, crop=True)
    valid_set = get_dataset(name=opt.dataset, root="./data", train=False)
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

    # define a two-layer noise-conditional MLP classifier
    feat_func = partial(model.get_feature, norm=False, use_amp=use_amp)
    with torch.no_grad():
        x = feat_func(next(iter(valid_loader))[0].to(device), t=0)
    print0("All block names:", x.keys())
    print0("Using block:", blockname)

    dim = x[blockname].shape[-1]
    model = Classifier(feat_func, blockname, dim, opt.classes).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)

    # train classifier
    loss_fn = nn.CrossEntropyLoss()
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (base_lr, DDP_multiplier))
    base_lr *= DDP_multiplier
    optim = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = CosineAnnealingLR(optim, epoch)
    for e in range(epoch):
        sampler.set_epoch(e)
        pbar = tqdm(train_loader, disable=(local_rank!=0))
        for i, (image, label) in enumerate(pbar):
            model.train()
            logit = model(*diff.train(image))
            label = label.to(device)
            loss = loss_fn(logit, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # logging
            dist.barrier()
            loss = reduce_tensor(loss)
            logit = gather_tensor(logit).cpu()
            label = gather_tensor(label).cpu()

            if local_rank == 0:
                pred = logit.argmax(dim=-1)
                acc = (pred == label).sum().item() / len(label)
                nowlr = optim.param_groups[0]['lr']
                pbar.set_description("[epoch %d / iter %d]: lr %.1e loss: %.3f, acc: %.3f" % (e, i, nowlr, loss.item(), acc))
        scheduler.step()

    accs = {}
    for t in diff.test_timesteps:
        test_acc = test(t)
        print0("[timestep %d]: Test acc: %.3f" % (t, test_acc))
        accs[t] = test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    print0(opt)

    local_rank = opt.local_rank
    init_seeds(no=local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank

    train(opt)
