import keras

from bayesflow.types import Tensor


def jvp(fn: callable, primals: tuple[Tensor] | Tensor, tangents: tuple[Tensor] | Tensor):
    """Jacobian v"""
    match keras.backend.backend():
        case "torch":
            import torch

            fn_output, cos_sin_dFdt = torch.autograd.functional.jvp(fn, primals, tangents)
        case "tensorflow":
            import tensorflow as tf

            with tf.autodiff.ForwardAccumulator(primals=primals, tangents=tangents) as acc:
                fn_output = fn(*primals)
            jvp = acc.jvp(fn_output)
        case "jax":
            import jax

            fn_output, cos_sin_dFdt = jax.jvp(
                fn,
                primals,
                tangents,
            )
        case _:
            raise NotImplementedError(f"JVP not implemented for backend {keras.backend.backend()}")
    return fn_output,
