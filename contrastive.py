import argparse
import os

import torch
import torch.distributed as dist
import yaml
from torchvision import transforms
from torchvision.datasets import CIFAR10
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
    print("loading model at", target)
    checkpoint = torch.load(target, map_location=device)
    ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
    ema.to(device)
    ema.load_state_dict(checkpoint['EMA'])
    model = ema.ema_model
    model.eval()
    return model


def alignment(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean().item()

def uniformity(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log().item()


class NamedMeter:
    def __init__(self):
        self.sum = {}
        self.count = {}
        self.history = {}

    def update(self, name, val, n=1):
        if name not in self.sum:
            self.sum[name] = 0
            self.count[name] = 0
            self.history[name] = []

        self.sum[name] += val * n
        self.count[name] += n
        self.history[name].append("%.4f" % val)

    def get_avg(self, name):
        return self.sum[name] / self.count[name]

    def get_names(self):
        return self.sum.keys()


def metrics(opt):
    yaml_path = opt.config
    interval = opt.epoch_interval
    use_amp = opt.use_amp
    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    timestep = opt.linear['timestep']

    train_set_raw = CIFAR10("./data", train=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    train_loader_raw, _ = DataLoaderDDP(
        train_set_raw,
        batch_size=128,
        shuffle=False,
    )

    check_epochs = list(range(interval, opt.n_epoch, interval)) + [opt.n_epoch - 1]
    align_evolving = NamedMeter()
    uniform_evolving = NamedMeter()
    
    print0("Using timestep =", timestep)
    print0("Checking epochs:", check_epochs)

    for load_epoch in check_epochs:
        model = get_model(opt, load_epoch)
        align_cur_epoch = NamedMeter()
        uniform_cur_epoch = NamedMeter()

        for image, _ in tqdm(train_loader_raw, disable=(local_rank!=0)):
            with torch.no_grad():
                x = model.get_feature(image.to(device), timestep, norm=True, use_amp=use_amp)
                y = model.get_feature(image.to(device), timestep, norm=True, use_amp=use_amp)
            dist.barrier()
            x = {name: gather_tensor(x[name]).cpu() for name in x}
            y = {name: gather_tensor(y[name]).cpu() for name in y}

            for blockname in x:
                align = alignment(x[blockname].detach(), y[blockname].detach())
                uniform = (uniformity(x[blockname]) + uniformity(y[blockname])) / 2
                # calculate metrics for a small batch
                align_cur_epoch.update(blockname, align, n=image.shape[0])
                uniform_cur_epoch.update(blockname, uniform, n=image.shape[0])

        # gather metrics for the complete dataset
        for blockname in align_cur_epoch.get_names():
            align = align_cur_epoch.get_avg(blockname)
            uniform = uniform_cur_epoch.get_avg(blockname)
            # record metrics for each checkpoint
            align_evolving.update(blockname, align)
            uniform_evolving.update(blockname, uniform)

    if local_rank == 0:
        print(align_evolving.history.keys())
        print('align metric:')
        for blockname in align_evolving.history:
            align = align_evolving.history[blockname]
            print("'%s': [%s]" % (blockname, ', '.join(align)))

        print('uniform metric:')
        for blockname in uniform_evolving.history:
            uniform = uniform_evolving.history[blockname]
            print("'%s': [%s]" % (blockname, ', '.join(uniform)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--epoch_interval', type=int, default=400)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    print(opt)

    local_rank = opt.local_rank
    init_seeds(no=local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank

    metrics(opt)
