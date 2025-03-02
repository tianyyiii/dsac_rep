"""
Encoders more suitable for ViT architectures.
- SmallStem: 3 conv layers, then patchifies the image (from xiao et al. 2021)
"""

import functools as ft
from typing import Callable, Sequence, TypeVar, Tuple
from dataclasses import dataclass

# from flax import linen as nn
import jax
import jax.numpy as jnp
import haiku as hk
from relax.utils.jax_utils import fix_repr



T = TypeVar("T")

def normalize_images(img, img_norm_type="default"):
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.astype(jnp.float32) / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0,1]
        img = img.astype(jnp.float32) / 255
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = jnp.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
        std = jnp.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = jnp.tile(mean, num_tile)
        std_tile = jnp.tile(std, num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile) / std_tile
    raise ValueError()


def weight_standardize(w, axis, eps):
    """Subtracts mean and divides by standard deviation."""
    w = w - jnp.mean(w, axis=axis)
    w = w / (jnp.std(w, axis=axis) + eps)
    return w


# class FilmConditioning(nn.Module):
#     @nn.compact
#     def __call__(self, conv_filters: jnp.ndarray, conditioning: jnp.ndarray):
#         """Applies FiLM conditioning to a convolutional feature map.

#         Args:
#             conv_filters: A tensor of shape [batch_size, height, width, channels].
#             conditioning: A tensor of shape [batch_size, conditioning_size].

#         Returns:
#             A tensor of shape [batch_size, height, width, channels].
#         """
#         projected_cond_add = nn.Dense(
#             features=conv_filters.shape[-1],
#             kernel_init=nn.initializers.zeros,
#             bias_init=nn.initializers.zeros,
#         )(conditioning)
#         projected_cond_mult = nn.Dense(
#             features=conv_filters.shape[-1],
#             kernel_init=nn.initializers.zeros,
#             bias_init=nn.initializers.zeros,
#         )(conditioning)

#         projected_cond_add = projected_cond_add[:, None, None, :]
#         projected_cond_mult = projected_cond_mult[:, None, None, :]

#         return conv_filters * (1 + projected_cond_add) + projected_cond_mult


# class StdConv(nn.Conv):
#     """Convolution with weight standardization."""

#     def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
#         param = super().param(name, init_fn, *init_args)
#         if name == "kernel":
#             param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
#         return param


# class SmallStem(nn.Module):
#     """Passes the image through a few light-weight convolutional layers,
#     before patchifying the image. Empirically useful for many computer vision tasks.

#     See Xiao et al: Early Convolutions Help Transformers See Better
#     """

#     use_film: bool = False
#     patch_size: int = 32
#     kernel_sizes: tuple = (3, 3, 3, 3)
#     strides: tuple = (2, 2, 2, 2)
#     features: tuple = (32, 96, 192, 384)
#     padding: tuple = (1, 1, 1, 1)
#     num_features: int = 512
#     img_norm_type: str = "default"

#     @nn.compact
#     def __call__(self, observations: jnp.ndarray, train: bool = True, cond_var=None):
#         expecting_cond_var = self.use_film
#         received_cond_var = cond_var is not None
#         assert (
#             expecting_cond_var == received_cond_var
#         ), "Only pass in cond var iff model expecting cond var"

#         x = normalize_images(observations, self.img_norm_type)
#         for n, (kernel_size, stride, features, padding) in enumerate(
#             zip(
#                 self.kernel_sizes,
#                 self.strides,
#                 self.features,
#                 self.padding,
#             )
#         ):
#             x = StdConv(
#                 features=features,
#                 kernel_size=(kernel_size, kernel_size),
#                 strides=(stride, stride),
#                 padding=padding,
#             )(x)
#             x = nn.GroupNorm()(x)
#             x = nn.relu(x)

#         x = nn.Conv(
#             features=self.num_features,
#             kernel_size=(self.patch_size // 16, self.patch_size // 16),
#             strides=(self.patch_size // 16, self.patch_size // 16),
#             padding="VALID",
#             name="embedding",
#         )(x)
#         if self.use_film:
#             assert cond_var is not None, "Cond var is None, nothing to condition on"
#             x = FilmConditioning()(x, cond_var)
#         return x


# class SmallStem16(SmallStem):
#     patch_size: int = 16


# class SmallStem32(SmallStem):
#     patch_size: int = 32



def max_pool(x: jnp.ndarray, kernel_shape: Tuple[int, int], strides: Tuple[int, int], padding: str):
    window_shape = (1,) + kernel_shape + (1,)
    stride_shape = (1,) + strides + (1,)
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window_shape, stride_shape, padding)

@dataclass
@fix_repr
class ResidualBlock(hk.Module):
    def __init__(self, filters: int, strides: Tuple[int, int] = (1, 1), name: str = None):
        super().__init__(name=name)
        self.filters = filters
        self.strides = strides

    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        residual = x

        x = hk.Conv2D(output_channels=self.filters,
                      kernel_shape=3,
                      stride=self.strides,
                      padding="SAME")(x)
        x = hk.LayerNorm(axis=[-1], create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)

        x = hk.Conv2D(output_channels=self.filters,
                      kernel_shape=3,
                      stride=1,
                      padding="SAME")(x)
        x = hk.LayerNorm(axis=[-1], create_scale=True, create_offset=True)(x)

        if residual.shape != x.shape:
            residual = hk.Conv2D(output_channels=self.filters,
                                 kernel_shape=1,
                                 stride=self.strides,
                                 padding="SAME")(residual)
            residual = hk.LayerNorm(axis=[-1], create_scale=True, create_offset=True)(residual)

        return jax.nn.relu(x + residual)

@dataclass
@fix_repr
class ResNetEncoder(hk.Module):
    def __init__(self, embedding_dim: int = 256, name: str = None):
        super().__init__(name=name)
        self.embedding_dim = embedding_dim

    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        x = hk.Conv2D(output_channels=64,
                      kernel_shape=7,
                      stride=2,
                      padding="SAME")(x)
        x = hk.LayerNorm(axis=[-1], create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)

        x = hk.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        x = ResidualBlock(64, strides=(1, 1))(x, is_training=is_training)
        x = ResidualBlock(64, strides=(1, 1))(x, is_training=is_training)

        x = ResidualBlock(128, strides=(2, 2))(x, is_training=is_training)
        x = ResidualBlock(128, strides=(1, 1))(x, is_training=is_training)

        x = ResidualBlock(256, strides=(2, 2))(x, is_training=is_training)
        x = ResidualBlock(256, strides=(1, 1))(x, is_training=is_training)

        x = jnp.mean(x, axis=(1, 2))
        if x.shape[-1] != self.embedding_dim:
            x = hk.Linear(self.embedding_dim)(x)
        return x
    

class ConvNetEncoder(hk.Module):
    def __init__(self, embedding_dim, name=None):
        super().__init__(name=name)

        self.repr_dim = 32 * 25 * 25
        
        self.convnet = hk.Sequential([
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=2, padding='VALID'),
            jax.nn.relu,
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='VALID'),
            jax.nn.relu,
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='VALID'),
            jax.nn.relu,
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='VALID'),
            jax.nn.relu,
        ])
        self.linear = hk.Linear(embedding_dim)

    def __call__(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = jnp.reshape(h, (h.shape[0], -1))
        h = self.linear(h)
        return h




# vit_encoder_configs = {
#     "small-stem-8-film": ft.partial(
#         SmallStem,
#         use_film=True,
#         patch_size=16,
#         kernel_sizes=(3, 3, 3),
#         strides=(2, 2, 2),
#         features=(32, 96, 192),
#         padding=(1, 1, 1),
#     ),
#     "small-stem-16": ft.partial(
#         SmallStem,
#         patch_size=16,
#     ),
#     "small-stem-16-film": ft.partial(
#         SmallStem,
#         use_film=True,
#         patch_size=16,
#     ),
#     "small-stem-32-film": ft.partial(
#         SmallStem,
#         use_film=True,
#         patch_size=32,
#     ),
# }