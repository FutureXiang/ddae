# Denoising Diffusion Autoencoders (DDAE)

<p align="center">
  <img src="https://github.com/FutureXiang/ddae/assets/33350017/b0825947-e58f-4c5e-b672-ec59465ac14d" width="480">
</p>

This is a multi-gpu PyTorch implementation of the paper [Denoising Diffusion Autoencoders are Unified Self-supervised Learners](https://arxiv.org/abs/2303.09769):
```bibtex
@inproceedings{ddae2023,
  title={Denoising Diffusion Autoencoders are Unified Self-supervised Learners},
  author={Xiang, Weilai and Yang, Hongyu and Huang, Di and Wang, Yunhong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

This repo contains:
- [x] Pre-training, sampling and FID evaluation code for diffusion models, including
  - Frameworks:
    - [x] DDPM & DDIM
    - [x] EDM (without augmentation)
    - [ ] Data augmentation pipeline proposed by EDM
  - Networks:
    - [x] The basic 35.7M DDPM UNet
    - [x] A larger 56M DDPM++ UNet
  - Datasets:
    - [x] CIFAR-10
    - [ ] Tiny-ImageNet
- [x] Feature quality evaluation code, including
  - [x] Linear probing and grid searching
  - [x] Contrastive metrics, i.e., alignment and uniformity
- [x] Noise-conditional classifier training and evaluation, including
  - [x] MLP classifier based on DDPM/EDM features
  - [x] WideResNet with VP/VE perturbation

## Acknowledgments
This repository is built on numerous open-source codebases such as [DDPM](https://github.com/hojonathanho/diffusion), [DDPM-pytorch](https://github.com/pesser/pytorch_diffusion), [DDIM](https://github.com/ermongroup/ddim), [EDM](https://github.com/NVlabs/edm), [Score-based SDE](https://github.com/yang-song/score_sde), and [align_uniform](https://github.com/SsnL/align_uniform).
