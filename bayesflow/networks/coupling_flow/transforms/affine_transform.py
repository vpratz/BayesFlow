import math

import keras.ops as ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from .transform import Transform


@serializable(package="networks.coupling_flow")
class AffineTransform(Transform):
    def __init__(self, clamp: float | None = 1.9, **kwargs):
        super().__init__(**kwargs)
        self.clamp = clamp

    @property
    def params_per_dim(self):
        return 2

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        scale, shift = ops.split(parameters, 2, axis=-1)

        return {"scale": scale, "shift": shift}

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        scale = parameters["scale"]

        # soft clamp
        if self.clamp is not None:
            (2.0 * self.clamp / math.pi) * ops.arctan(scale / self.clamp)

        # constrain to positive values
        scale = ops.exp(scale)

        parameters["scale"] = scale
        return parameters

    def _forward(self, x: Tensor, parameters: dict[str, Tensor] = None) -> (Tensor, Tensor):
        z = parameters["scale"] * x + parameters["shift"]
        log_det = ops.sum(ops.log(parameters["scale"]), axis=-1)

        return z, log_det

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor] = None) -> (Tensor, Tensor):
        x = (z - parameters["shift"]) / parameters["scale"]
        log_det = -ops.sum(ops.log(parameters["scale"]), axis=-1)

        return x, log_det
