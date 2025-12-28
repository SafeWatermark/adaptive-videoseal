from videoseal.modules.msg_processor import MsgProcessor
from msg_processors.base_layer import MsgLayer
import torch

VIDEOSEAL_V1_MSG_PROCESSOR_HIDDEN_SIZE = 256


class LayeredMsgProcessor(MsgProcessor):
    """
    MsgProcessor that generates base message bits via super().get_random_msg()
    then applies a configurable list of layers.
    """

    def __init__(
        self,
        nbits: int,
        hidden_size: int = VIDEOSEAL_V1_MSG_PROCESSOR_HIDDEN_SIZE,
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0,
        layers: list[MsgLayer] | None = None,
    ):
        super().__init__(nbits, hidden_size, msg_processor_type, msg_mult)
        self._layers: list[MsgLayer] = list(layers) if layers is not None else []

    def add_layer(self, layer: MsgLayer) -> None:
        self._layers.append(layer)

    def remove_layer(self, layer_type: type) -> None:
        """Remove the first layer instance matching a type."""
        for i, layer in enumerate(self._layers):
            if isinstance(layer, layer_type):
                self._layers.pop(i)
                return

    def clear_layers(self) -> None:
        self._layers = []

    def list_layers(self) -> list[str]:
        return [layer.name for layer in self._layers]

    def get_random_msg(self, bsz: int = 1, nb_repetitions: int = 1) -> torch.Tensor:
        msg = super().get_random_msg(bsz, nb_repetitions)

        for layer in self._layers:
            msg = layer(msg)

        return msg
