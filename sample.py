import argparse
import os

import torch
import torch.distributed as dist
import yaml
from torchvision.utils import make_grid, save_image
from ema_pytorch import EMA

from model.models import get_models_class
from utils import Config, init_seeds, gather_tensor, print0


def get_default_steps(model_type, steps):
    if steps is not None:
        return steps
    else:
        return {'DDPM': 100, 'EDM': 18}[model_type]


# ===== sampling =====

def sample(opt):
    print0(opt)
    yaml_path = opt.config
    local_rank = opt.local_rank
    use_amp = opt.use_amp
    mode = opt.mode
    steps = opt.steps
    eta = opt.eta
    batches = opt.batches
    ep = opt.epoch

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    if ep == -1:
        ep = opt.n_epoch - 1

    device = "cuda:%d" % local_rank
    steps = get_default_steps(opt.model_type, steps)
    DIFFUSION, NETWORK = get_models_class(opt.model_type, opt.net_type)
    diff = DIFFUSION(nn_model=NETWORK(**opt.network),
                     **opt.diffusion,
                     device=device,
                     )
    diff.to(device)

    target = os.path.join(opt.save_dir, "ckpts", f"model_{ep}.pth")
    print0("loading model at", target)
    checkpoint = torch.load(target, map_location=device)
    ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
    ema.to(device)
    ema.load_state_dict(checkpoint['EMA'])
    model = ema.ema_model
    model.eval()

    if local_rank == 0:
        if opt.model_type == 'EDM':
            gen_dir = os.path.join(opt.save_dir, f"EMAgenerated_ep{ep}_edm_steps{steps}_eta{eta}")
        else:
            if mode == 'DDPM':
                gen_dir = os.path.join(opt.save_dir, f"EMAgenerated_ep{ep}_ddpm")
            else:
                gen_dir = os.path.join(opt.save_dir, f"EMAgenerated_ep{ep}_ddim_steps{steps}_eta{eta}")
        os.makedirs(gen_dir)
        gen_dir_png = os.path.join(gen_dir, "pngs")
        os.makedirs(gen_dir_png)
        res = []

    for batch in range(batches):
        with torch.no_grad():
            assert 400 % dist.get_world_size() == 0
            samples_per_process = 400 // dist.get_world_size()
            args = dict(n_sample=samples_per_process, size=opt.network['image_shape'], notqdm=(local_rank != 0), use_amp=use_amp)
            if opt.model_type == 'EDM':
                x_gen = model.edm_sample(**args, steps=steps, eta=eta)
            else:
                if mode == 'DDPM':
                    x_gen = model.sample(**args)
                else:
                    x_gen = model.ddim_sample(**args, steps=steps, eta=eta)
        dist.barrier()
        x_gen = gather_tensor(x_gen)
        if local_rank == 0:
            res.append(x_gen)
            grid = make_grid(x_gen.cpu(), nrow=20)
            png_path = os.path.join(gen_dir, f"grid_{batch}.png")
            save_image(grid, png_path)

    if local_rank == 0:
        res = torch.cat(res)
        for no, img in enumerate(res):
            png_path = os.path.join(gen_dir_png, f"{no}.png")
            save_image(img, png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument("--mode", type=str, choices=['DDPM', 'DDIM'], default='DDIM')
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--batches", type=int, default=125)
    parser.add_argument("--epoch", type=int, default=-1)
    opt = parser.parse_args()

    init_seeds(no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    sample(opt)
