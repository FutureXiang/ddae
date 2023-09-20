from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast


def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


def linear_beta_schedule(timesteps, beta1, beta2):
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    return torch.linspace(beta1, beta2, timesteps)


def schedules(betas, T, device, type='DDPM'):
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


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        ''' DDPM proposed by "Denoising Diffusion Probabilistic Models", and \
            DDIM sampler proposed by "Denoising Diffusion Implicit Models".

            Args:
                nn_model: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Parameters:
                betas, n_T
        '''
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6
        print(f"nn model # params: {params:.1f}")

        self.device = device
        self.ddpm_sche = schedules(betas, n_T, device, 'DDPM')
        self.ddim_sche = schedules(betas, n_T, device, 'DDIM')
        self.n_T = n_T
        self.loss = nn.MSELoss()

    def perturb(self, x, t=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, n_T]`. Type: int / torch.LongTensor / None. \
                    Random `t ~ U[1, n_T]` is taken if t is None.
            Returns:
                The perturbed image, the corresponding timestep, and the noise.
        '''
        if t is None:
            t = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][t, None, None, None] * x +
                    sche["sqrtmab"][t, None, None, None] * noise)
        return x_noised, t, noise

    def forward(self, x, use_amp=False):
        ''' Training with simple noise prediction loss.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
            Returns:
                The simple MSE loss.
        '''
        x = normalize_to_neg_one_to_one(x)
        x_noised, t, noise = self.perturb(x, t=None)

        with autocast(enabled=use_amp):
            return self.loss(noise, self.nn_model(x_noised, t / self.n_T))

    def get_feature(self, x, t, name=None, norm=False, use_amp=False):
        ''' Get network's intermediate activation in a forward pass.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
                t: The specified timestep ranged in `[1, n_T]`. Type: int / torch.LongTensor.
                norm: to normalize features to the the unit hypersphere.
            Returns:
                A {name: tensor} dict which contains global average pooled features.
        '''
        x = normalize_to_neg_one_to_one(x)
        x_noised, t, noise = self.perturb(x, t)

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

        with autocast(enabled=use_amp):
            _, acts = self.nn_model(x_noised, t / self.n_T, ret_activation=True)
        all_feats = {blockname: gap_and_norm(acts[blockname], norm) for blockname in acts}
        if name is not None:
            return all_feats[name]
        else:
            return all_feats

    def sample(self, n_sample, size, notqdm=False, use_amp=False):
        ''' Sampling with DDPM sampler. Actual NFE is `n_T`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddpm_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        for i in tqdm(range(self.n_T, 0, -1), disable=notqdm):
            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0

            alpha = sche["alphabar_t"][i]
            eps, _ = self.pred_eps_(x_i, t_is, alpha, use_amp)

            mean = sche["oneover_sqrta"][i] * (x_i - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            x_i = mean + variance * z

        return unnormalize_to_zero_to_one(x_i)

    def ddim_sample(self, n_sample, size, steps=100, eta=0.0, notqdm=False, use_amp=False):
        ''' Sampling with DDIM sampler. Actual NFE is `steps`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # e.g. [(801, 601), (601, 401), (401, 201), (201, 1), (1, 0)]

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            t_is = torch.tensor([time / self.n_T]).to(self.device).repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if time_next > 0 else 0

            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha, use_amp)
            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()
            x_i = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

        return unnormalize_to_zero_to_one(x_i)

    def pred_eps_(self, x, t, alpha, use_amp, clip_x=True):
        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        # get prediction of x0
        with autocast(enabled=use_amp):
            eps = self.nn_model(x, t).float()
        denoised = pred_x0_from_eps(eps)

        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised
