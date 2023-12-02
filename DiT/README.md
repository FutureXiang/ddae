## DDAE/DiT

This subfolder contains transfer learning evaluation for ImageNet-256 pre-trained [DiT-XL/2](https://github.com/facebookresearch/DiT) checkpoint, by:
- evaluating:
  - [x] Linear probing
  - [ ] Fine-tuning
- performance on these datasets:
  - [x] CIFAR-10
  - [x] Tiny-ImageNet

This implementation uses very small batch sizes, lightweight data augmentations, and a standard Adam optimizer, without advanced optimizer (e.g., LARS) and large batch sizes. However, incorporating these modern tricks may further improve performances.

## Main results
The pre-trained DiT-XL/2 is expected to achieve $85.73$ % linear probing accuracy on CIFAR-10, and $66.57$ % on Tiny-ImageNet.

## Usage
### Data pre-processing
Since DiT is operating in the latent-space, we need to resize the images to $256\times256$ and generate their latent codes (shape: $(4,32,32)$ ) through the VAE encoder.

To reduce the computational cost at the training, we use `vae_preprocessing.py` to pre-calculate and cache the latent codes into files. Since data augmentations are essential for effective discriminative learning, we generate multiple versions (by default, 10) of latent codes to cover different variations of augmented images. Please refer to `vae_preprocessing.py` for more details.

```sh
python -m torch.distributed.launch --nproc_per_node=4
  # pre-processing with VAE encoding
  vae_preprocessing.py --dataset cifar --use_amp
  vae_preprocessing.py --dataset tiny  --use_amp
```

### Linear probing
To linear probe the features produced by pre-trained DiT, for example, run:
```sh
python -m torch.distributed.launch --nproc_per_node=4
  # linear probing with default layer-noise combination
  linear.py --dataset cifar --use_amp
  linear.py --dataset tiny  --use_amp
```
Note that this implementation loads ALL versions of the augmented dataset (by default, 10) into the memory, and hence it requires A LOT OF memory to run (e.g., 50 GB for CIFAR, 80GB for Tiny-ImageNet).
You can improve this by dumping each latent code into a standalone numpy file and only load it when needed, in case you don't have enough memory to work with.

## Acknowledgments
Except for `vae_preprocessing.py` and `linear.py`, all codes are retrieved or modified from the official [DiT](https://github.com/facebookresearch/DiT) repository.
