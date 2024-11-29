from collections.abc import Sequence

from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transform import Transform


@serializable(package="bayesflow.adapters")
class Keep(Transform):
    '''
    Name the data parameters that should be kept for futher calculation. 

    Parameters: 

        cls: tuple containing the names of kept data variables as strings. 

    Example: 
        Two moons simulator generates data for priors alpha, r and theta as well as observation data x.
        We are interested only in theta and x, to keep only theta and x we should use the following; 

        adapter = (
            bf.adapters.Adapter()

            # drop data from unneeded priors alpha, and r 
            .keep(("theta", "x"))

            )


    '''
    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Keep":
        return cls(keys=deserialize(config["keys"], custom_objects))

    def get_config(self) -> dict:
        return {"keys": serialize(self.keys)}

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        return {key: value for key, value in data.items() if key in self.keys}

    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        # non-invertible transform
        return data

    def extra_repr(self) -> str:
        return "[" + ", ".join(map(repr, self.keys)) + "]"
