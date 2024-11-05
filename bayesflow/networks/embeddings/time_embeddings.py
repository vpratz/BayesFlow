import keras
from keras import ops

import numpy as np


class GaussianFourierEmbedding(keras.layers.Layer):
    """Fourier projection with normally distributed frequencies"""

    def __init__(self, fourier_emb_dim, scale=1.0, include_identity=True):
        """Create an instance of a fourier projection with normally
        distributed frequencies.
        Parameters:
        -----------
        fourier_emb_dim   : int (even)
            Dimensionality of the fourier projection. The complete embedding has
            dimensionality `fourier_embed_dim + 1` if the identity mapping is
            added as well.

        """
        super().__init__()
        assert fourier_emb_dim % 2 == 0, f"Embedding dimension must be even, was {fourier_emb_dim}."
        self.w = self.add_weight(initializer="random_normal", shape=(fourier_emb_dim // 2,), trainable=False)
        self.scale = scale
        self.include_identity = include_identity

    def call(self, t):
        """
        Parameters:
        -----------
        t   : Tensor of shape (batch_size, 1)
            vector of times

        Returns:
        --------
        emb : Tensor
            Embedding of shape (batch_size, fourier_emb_dim) if `include_identity`
            is False, else (batch_size, fourier_emb_dim+1)
        """
        proj = t * self.w[None, :] * 2 * np.pi * self.scale
        if self.include_identity:
            emb = ops.concatenate([t, ops.sin(proj), ops.cos(proj)], axis=-1)
        else:
            emb = ops.concatenate([ops.sin(proj), ops.cos(proj)], axis=-1)
        return emb
