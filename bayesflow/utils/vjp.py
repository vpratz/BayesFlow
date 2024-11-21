import keras

from bayesflow.types import Tensor


def vjp(f: callable, x: Tensor) -> (Tensor, callable):
    match keras.backend.backend():
        case "jax":
            import jax

            fx, _vjp_fn = jax.vjp(f, x)

            def vjp_fn(projector):
                return _vjp_fn(projector)[0]
        case "torch":
            import torch

            fx, _vjp_fn = torch.func.vjp(f, x)

            def vjp_fn(projector):
                return _vjp_fn(projector)[0]
        case "tensorflow":
            import tensorflow as tf

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                fx = f(x)

            def vjp_fn(projector):
                return tape.gradient(fx, x, projector)
        case other:
            raise NotImplementedError(f"Cannot build a vjp function for backend '{other}'.")

    return fx, vjp_fn
