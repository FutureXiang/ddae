import argparse
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import get_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.wideresnet_noise_song import wide_28_10_song
from utils import init_seeds, reduce_tensor, gather_tensor, DataLoaderDDP, print0


def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


class DDPM:
    def __init__(self, device, betas=[1.0e-4, 0.02], n_T=1000, steps=20):
        self.device = device
        self.n_T = n_T
        self.ddpm_sche = self.schedules(betas, n_T, device, 'DDPM')
        self.test_timesteps = (torch.arange(0, self.n_T, self.n_T // steps) + 1).long().tolist()

    def train(self, x):
        x = normalize_to_neg_one_to_one(x).to(self.device)
        # Perturbation
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][_ts, None, None, None] * x +
                    sche["sqrtmab"][_ts, None, None, None] * noise)
        return x_noised, _ts / self.n_T

    def test(self, x, t):
        x = normalize_to_neg_one_to_one(x).to(self.device)
        # Perturbation
        _ts = torch.tensor([t]).to(self.device).repeat(x.shape[0])
        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][_ts, None, None, None] * x +
                    sche["sqrtmab"][_ts, None, None, None] * noise)
        return x_noised, _ts / self.n_T

    def schedules(self, betas, T, device, type='DDPM'):
        def linear_beta_schedule(timesteps, beta1, beta2):
            assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
            return torch.linspace(beta1, beta2, timesteps)

        beta1, beta2 = betas
        schedule_fn = partial(linear_beta_schedule, beta1=beta1, beta2=beta2)

        if type == 'DDPM':
            beta_t = torch.cat([torch.tensor([0.0]), schedule_fn(T)])
        elif type == 'DDIM':
            beta_t = schedule_fn(T + 1)
        else:
            raise NotImplementedError()
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

        dic = {
            "alpha_t": alpha_t,
            "oneover_sqrta": oneover_sqrta,
            "sqrt_beta_t": sqrt_beta_t,
            "alphabar_t": alphabar_t,
            "sqrtab": sqrtab,
            "sqrtmab": sqrtmab,
            "ma_over_sqrtmab": ma_over_sqrtmab,
        }
        return {key: dic[key].to(device) for key in dic}


class EDM:
    def __init__(self, device, p_std=1.2, p_mean=-1.2, sigma_min=0.002, sigma_max=80, rho=7, steps=18):
        self.device = device
        self.p_std = p_std
        self.p_mean = p_mean
        self.times = self.schedules(sigma_min, sigma_max, rho, steps)
        self.test_timesteps = range(1, steps + 1)

    def train(self, x):
        x = normalize_to_neg_one_to_one(x).to(self.device)
        # Perturbation
        rnd_normal = torch.randn((x.shape[0], 1, 1, 1)).to(self.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        noise = torch.randn_like(x)
        x_noised = x + noise * sigma
        
        sigma = sigma.reshape(x.shape[0],)
        return x_noised, sigma.log()

    def test(self, x, t):
        x = normalize_to_neg_one_to_one(x).to(self.device)
        # Perturbation
        noise = torch.randn_like(x)
        sigma = self.times[t]
        x_noised = x + noise * sigma

        sigma = torch.full((x.shape[0], ), sigma)
        return x_noised, sigma.log()

    def schedules(self, sigma_min, sigma_max, rho, steps):
        times = torch.arange(steps, dtype=torch.float64, device=self.device)
        times = (sigma_max ** (1 / rho) + times / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        times = torch.cat([times, torch.zeros_like(times[:1])]) # t_N = 0
        times = reversed(times)
        return times


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

    warm_epoch = opt.warm_epoch
    epoch = opt.epoch
    batch_size = opt.batch_size
    base_lr = opt.lr
    mode = opt.mode

    if mode == 'DDPM':
        diff = DDPM(device)
    elif mode == 'EDM':
        diff = EDM(device)
    else:
        raise NotImplementedError

    train_set = get_dataset(name='cifar', root="./data", train=True, flip=True, crop=True)
    valid_set = get_dataset(name='cifar', root="./data", train=False)
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

    model = wide_28_10_song(num_classes=10).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optim, epoch)
    for e in range(epoch):
        sampler.set_epoch(e)
        if (e + 1) <= warm_epoch:
            for g in optim.param_groups:
                g['lr'] = base_lr * (e + 1.0) / warm_epoch # warmup

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
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--warm_epoch', default=5, type=int)
    parser.add_argument("--mode", type=str, choices=['DDPM', 'EDM'], default='DDPM')
    opt = parser.parse_args()
    print0(opt)

    local_rank = opt.local_rank
    init_seeds(no=local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank

    train(opt)
