import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from .augment import AugmentPipe


def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


class EDM(nn.Module):
    def __init__(self, nn_model,
                 sigma_data, p_mean, p_std,
                 sigma_min, sigma_max, rho,
                 S_min, S_max, S_noise,
                 device,
                 augment_prob=0):
        ''' EDM proposed by "Elucidating the Design Space of Diffusion-Based Generative Models".

            Args:
                nn_model: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Training parameters:
                sigma_data, p_mean, p_std
                augment_prob
            Sampling parameters:
                sigma_min, sigma_max, rho
                S_min, S_max, S_noise
        '''
        super(EDM, self).__init__()
        self.nn_model = nn_model.to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6
        print(f"nn model # params: {params:.1f}")

        self.device = device

        def number_to_torch_device(value):
            return torch.tensor(value).to(device)

        self.sigma_data = number_to_torch_device(sigma_data)
        self.p_mean     = number_to_torch_device(p_mean)
        self.p_std      = number_to_torch_device(p_std)
        self.sigma_min  = number_to_torch_device(sigma_min)
        self.sigma_max  = number_to_torch_device(sigma_max)
        self.rho        = number_to_torch_device(rho)
        self.S_min      = number_to_torch_device(S_min)
        self.S_max      = number_to_torch_device(S_max)
        self.S_noise    = number_to_torch_device(S_noise)
        if augment_prob > 0:
            self.augpipe = AugmentPipe(p=augment_prob, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        else:
            self.augpipe = None

    def perturb(self, x, t=None, steps=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, steps]`. Type: int / torch.LongTensor / None. \
                    Random `ln(sigma) ~ N(P_mean, P_std)` is taken if t is None.
            Returns:
                The perturbed image, and the corresponding sigma.
        '''
        if t is None:
            rnd_normal = torch.randn((x.shape[0], 1, 1, 1)).to(self.device)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        else:
            times = reversed(self.sample_schedule(steps))
            sigma = times[t]
            if len(sigma.shape) == 1:
                sigma = sigma[:, None, None, None]

        noise = torch.randn_like(x)
        x_noised = x + noise * sigma
        return x_noised, sigma

    def forward(self, x, use_amp=False):
        ''' Training with weighted denoising loss.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
            Returns:
                The weighted MSE loss.
        '''
        x = normalize_to_neg_one_to_one(x)
        x, aug_label = self.augpipe(x) if self.augpipe is not None else (x, None)
        x_noised, sigma = self.perturb(x, t=None)

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss_4shape = weight * ((x - self.D_x(x_noised, sigma, use_amp, aug_label)) ** 2)
        return loss_4shape.mean()

    def get_feature(self, x, t, steps=18, norm=False, use_amp=False):
        ''' Get network's intermediate activation in a forward pass.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
                t: The specified timestep ranged in `[1, steps]`. Type: int / torch.LongTensor.
                norm: to normalize features to the the unit hypersphere.
            Returns:
                A {name: tensor} dict which contains global average pooled features.
        '''
        x = normalize_to_neg_one_to_one(x)
        x_noised, sigma = self.perturb(x, t, steps)

        def gap_and_norm(act, norm=False):
            if len(act.shape) == 4:
                # unet (B, C, H, W)
                act = act.view(act.shape[0], act.shape[1], -1).float()
                act = torch.mean(act, dim=2)
            else:
                raise NotImplementedError
            if norm:
                act = torch.nn.functional.normalize(act)
            return act

        _, acts = self.D_x(x_noised, sigma, use_amp, ret_activation=True)
        return {blockname: gap_and_norm(acts[blockname], norm) for blockname in acts}

    def edm_sample(self, n_sample, size, steps=18, eta=0.0, notqdm=False, use_amp=False):
        ''' Sampling with EDM sampler. Actual NFE is `2 * steps - 1`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        S_min, S_max, S_noise = self.S_min, self.S_max, self.S_noise
        gamma_stochasticity = torch.tensor(np.sqrt(2) - 1) * eta # S_churn = (sqrt(2) - 1) * eta * steps

        times = self.sample_schedule(steps)
        time_pairs = list(zip(times[:-1], times[1:]))

        x_next = torch.randn(n_sample, *size).to(self.device).to(torch.float64) * times[0]
        for i, (t_cur, t_next) in enumerate(tqdm(time_pairs, disable=notqdm)): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = gamma_stochasticity if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            d_cur = self.pred_eps_(x_hat, t_hat, use_amp)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < steps - 1:
                d_prime = self.pred_eps_(x_next, t_next, use_amp)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return unnormalize_to_zero_to_one(x_next)

    def pred_eps_(self, x, t, use_amp, clip_x=True):
        denoised = self.D_x(x, t, use_amp).to(torch.float64)
        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
        eps = (x - denoised) / t
        return eps

    def D_x(self, x_noised, sigma, use_amp, aug_label=None, ret_activation=False):
        ''' Denoising with network preconditioning.

            Args:
                x_noised: The perturbed image tensor.
                sigma: The variance (noise level) tensor.
                aug_label: The augmentation labels produced by AugmentPipe.
            Returns:
                The estimated denoised image tensor.
                The {name: (B, C, H, W) tensor} activation dict (if ret_activation is True).
        '''
        x_noised = x_noised.to(torch.float32)
        sigma = sigma.to(torch.float32)

        # Preconditioning
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Denoising
        with autocast(enabled=use_amp):
            F_x = self.nn_model(c_in * x_noised, c_noise.flatten(), aug_label, ret_activation)

        if ret_activation:
            return c_skip * x_noised + c_out * F_x[0], F_x[1]
        else:
            return c_skip * x_noised + c_out * F_x

    def sample_schedule(self, steps):
        ''' Make the variance schedule for EDM sampling.

            Args:
                steps: The number of total timesteps. Typically 18, 50 or 100.
            Returns:
                times: A decreasing tensor list such that
                    `times[0] == sigma_max`,
                    `times[steps-1] == sigma_min`, and
                    `times[steps] == 0`.
        '''
        sigma_min, sigma_max, rho = self.sigma_min, self.sigma_max, self.rho
        times = torch.arange(steps, dtype=torch.float64, device=self.device)
        times = (sigma_max ** (1 / rho) + times / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        times = torch.cat([times, torch.zeros_like(times[:1])]) # t_N = 0
        return times
