import keras
from keras import ops
from keras.saving import (
    register_keras_serializable,
)

import numpy as np

from bayesflow.types import Tensor
from bayesflow.utils import find_network, keras_kwargs, expand_right_as, expand_right_to


from ..inference_network import InferenceNetwork
from ..embeddings import GaussianFourierEmbedding


@register_keras_serializable(package="bayesflow.networks")
class ContinuousConsistencyModel(InferenceNetwork):
    """Implements an sCM (simple, stable, and scalable Consistency Model)
    with continous-time Consistency Training (CT) as described in [1].
    The sampling procedure is taken from [2].

    [1] Lu, C., & Song, Y. (2024).
    Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models
    arXiv preprint arXiv:2410.11081

    [2] Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023).
    Consistency Models.
    arXiv preprint arXiv:2303.01469
    """

    def __init__(
        self,
        subnet: str | type = "mlp",
        sigma_data: float = 1.0,
        time_emb_dim: int = 20,
        **kwargs,
    ):
        """Creates an instance of an sCM to be used for consistency training (CT).

        Parameters:
        -----------
        subnet      : str or type, optional, default: "mlp"
            A neural network type for the consistency model, will be
            instantiated using subnet_kwargs.
        sigma_data      : float, optional, default: 1.0
            Standard deviation of the target distribution
        time_emb_dim  : int, optional, default: 20
            Dimensionality of a time embedding. The embedding will
            be concatenated to the time, so the total time embedding
            will have size `time_emb_dim + 1`
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments
        """
        # Normal is the only supported base distribution for CMs
        super().__init__(base_distribution="normal", **keras_kwargs(kwargs))

        self.subnet = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.subnet_projector = keras.layers.Dense(units=None, bias_initializer="zeros", kernel_initializer="zeros")

        self.weight_fn = find_network("mlp", widths=(256,), dropout=0.0)
        self.weight_fn_projector = keras.layers.Dense(units=1, bias_initializer="zeros", kernel_initializer="zeros")

        self.time_emb_dim = time_emb_dim
        self.time_emb = GaussianFourierEmbedding(self.time_emb_dim, scale=1.0, include_identity=True)

        self.sigma_data = sigma_data

        self.seed_generator = keras.random.SeedGenerator()

    def _discretize_time(self, num_steps, min_noise=0.001, max_noise=80.0, rho=7.0):
        """Function for obtaining the discretized time for multi-step sampling
        according to [2], Section 2, bottom of page 2.
        Subsequent transformation to time space following [1].
        """

        N = num_steps + 1
        indices = ops.arange(1, N + 1, dtype="float32")
        one_over_rho = 1.0 / rho
        discretized_time = (
            min_noise**one_over_rho
            + (indices - 1.0) / (ops.cast(N, "float32") - 1.0) * (max_noise**one_over_rho - min_noise**one_over_rho)
        ) ** rho
        time = ops.arctan(discretized_time / self.sigma_data)
        return time

    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)
        self.subnet_projector.units = xz_shape[-1]

        # construct input shape for subnet and subnet projector
        input_shape = list(xz_shape)

        # time vector
        input_shape[-1] += self.time_emb_dim + 1

        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.subnet.build(input_shape)

        input_shape = self.subnet.compute_output_shape(input_shape)
        self.subnet_projector.build(input_shape)

        # input shape for time embedding
        self.time_emb.build((xz_shape[0], 1))

        # input shape for weight function and projector
        input_shape = (xz_shape[0], 1)
        self.weight_fn.build(input_shape)
        input_shape = self.weight_fn.compute_output_shape(input_shape)
        self.weight_fn_projector.build(input_shape)

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        inverse: bool = False,
        **kwargs,
    ):
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

    def _forward(self, x: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        # Consistency Models only learn the direction from noise distribution
        # to target distribution, so we cannot implement this function.
        raise NotImplementedError("Consistency Models are not invertible")

    def _inverse(self, z: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        """Generate random draws from the approximate target distribution
        using the multistep sampling algorithm from [2], Algorithm 1.

        Parameters
        ----------
        z           : Tensor
            Samples from a standard normal distribution
        conditions  : Tensor, optional, default: None
            Conditions for a approximate conditional distribution
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments. Include `steps` (default: 30) to
            adjust the number of sampling steps.

        Returns
        -------
        x            : Tensor
            The approximate samples
        """
        steps = kwargs.get("steps", 30)
        max_noise = kwargs.get("max_noise", 80.0)
        min_noise = kwargs.get("min_noise", 1e-4)
        rho = kwargs.get("rho", 7.0)

        # noise distribution has variance sigma_data
        x = keras.ops.copy(z) * self.sigma_data
        discretized_time = keras.ops.flip(
            self._discretize_time(steps, max_noise=max_noise, min_noise=min_noise, rho=rho), axis=-1
        )
        t = keras.ops.full((*keras.ops.shape(x)[:-1], 1), np.pi / 2, dtype=x.dtype)
        x = self.consistency_function(x, t, conditions=conditions)
        for n in range(1, steps):
            noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator)
            x_n = ops.cos(t) * x + ops.sin(t) * noise
            t = keras.ops.full_like(t, discretized_time[n])
            x = self.consistency_function(x_n, t, conditions=conditions)
        return x

    def consistency_function(self, x: Tensor, t: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        """Compute consistency function.

        Parameters
        ----------
        x           : Tensor
            Input vector
        t           : Tensor
            Vector of time samples in [0, pi/2]
        conditions  : Tensor
            The conditioning vector
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the network.
        """

        if conditions is not None:
            xtc = ops.concatenate([x / self.sigma_data, self.time_emb(t), conditions], axis=-1)
        else:
            xtc = ops.concatenate([x / self.sigma_data, self.time_emb(t)], axis=-1)

        f = self.subnet_projector(self.subnet(xtc, **kwargs))

        out = ops.cos(t) * x - ops.sin(t) * self.sigma_data * f
        return out

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)

        # $# Implements Algorithm 1 from [1]

        # training parameters
        p_mean = -1.0
        p_std = 1.6

        c = 0.1

        # generate noise vector
        z = (
            keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator)
            * self.sigma_data
        )

        # sample time
        tau = (
            keras.random.normal(keras.ops.shape(x)[:1], dtype=keras.ops.dtype(x), seed=self.seed_generator) * p_std
            + p_mean
        )
        t_ = ops.arctan(ops.exp(tau) / self.sigma_data)
        t = expand_right_as(t_, x)

        # generate noisy sample
        xt = ops.cos(t) * x + ops.sin(t) * z

        # calculate estimator for dx_t/dt
        dxtdt = ops.cos(t) * z - ops.sin(t) * x

        r = 1.0  # TODO: if consistency distillation training (not supported yet) is unstable, add schedule here

        # calculate rearranged JVP
        if conditions is not None:

            def f_teacher(x, t):
                return self.subnet_projector(self.subnet(ops.concatenate([x, self.time_emb(t), conditions], axis=-1)))
        else:

            def f_teacher(x, t):
                return self.subnet_projector(self.subnet(ops.concatenate([x, self.time_emb(t)], axis=-1)))

        primals = (xt / self.sigma_data, t)
        tangents = (ops.cos(t) * ops.sin(t) * dxtdt, ops.cos(t) * ops.sin(t) * self.sigma_data)
        match keras.backend.backend():
            case "torch":
                import torch

                teacher_output, cos_sin_dFdt = torch.autograd.functional.jvp(f_teacher, primals, tangents)
            case "tensorflow":
                import tensorflow as tf

                with tf.autodiff.ForwardAccumulator(primals=primals, tangents=tangents) as acc:
                    teacher_output = f_teacher(xt / self.sigma_data, t)
                cos_sin_dFdt = acc.jvp(teacher_output)
            case "jax":
                import jax

                teacher_output, cos_sin_dFdt = jax.jvp(
                    f_teacher,
                    primals,
                    tangents,
                )
            case _:
                raise NotImplementedError(f"JVP not implemented for backend {keras.backend.backend()}")
        teacher_output = ops.stop_gradient(teacher_output)
        cos_sin_dFdt = ops.stop_gradient(cos_sin_dFdt)

        # calculate output of the network
        if conditions is not None:
            xtc = ops.concatenate([xt / self.sigma_data, self.time_emb(t), conditions], axis=-1)
        else:
            xtc = ops.concatenate([xt / self.sigma_data, self.time_emb(t)], axis=-1)
        student_out = self.subnet_projector(self.subnet(xtc, training=stage == "training"))

        # calculate the tangent
        g = -(ops.cos(t) ** 2) * (self.sigma_data * teacher_output - dxtdt) - r * (
            ops.cos(t) * ops.sin(t) * xt + self.sigma_data * cos_sin_dFdt
        )
        # apply normalization to stabilize training
        g = g / (ops.norm(g) + c)

        # compute adaptive weights
        w = self.weight_fn_projector(self.weight_fn(expand_right_to(t_, 2)))
        # calculate loss
        D = ops.shape(x)[-1]
        loss = ops.mean(
            (ops.exp(w) / D)
            * ops.mean(
                ops.reshape(((student_out - teacher_output - g) ** 2), (ops.shape(teacher_output)[0], -1)), axis=-1
            )
            - w
        )

        return base_metrics | {"loss": loss}
