import torch
from videoseal.models.embedder import Embedder
from ecc_codecs.rs_codec import RSWatermarkCodec
import numpy as np
import copy


class ECCEmbedder(Embedder):
    @classmethod
    def from_embedder(
        cls,
        model: Embedder,
        total_bits: int,
        message_bits: int,
        threshold: float = 0,
    ) -> "ECCEmbedder":
        new = copy.deepcopy(model)
        new.__class__ = cls

        new.total_bits = total_bits
        new.message_bits = message_bits
        new.codec = RSWatermarkCodec(total_bits, message_bits, threshold)
        new.register_forward_pre_hook(
            new._ecc_forward_pre_hook,
            with_kwargs=True,
        )
        new.forward = model.forward
        new.get_last_layer = model.get_last_layer

        return new

    def get_random_msg(self, bsz: int = 1, nb_repetitions: int = 1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.message_bits))

    def _apply_ecc_encoding(self, msgs: torch.Tensor) -> torch.Tensor:
        """Apply ECC encoding to message bits.

        Args:
            msgs: (B, message_bits) tensor of message bits.

        Returns:
            (B, self.total_bits) tensor with ECC-encoded bits.
        """
        device = msgs.device
        batch_size = msgs.shape[0]
        encoded_msgs = []

        for i in range(batch_size):
            msg_np = msgs[i].detach().cpu().numpy().astype(np.uint8)
            encoded = self.codec.encode(msg_np)
            encoded_msgs.append(encoded)

        encoded_tensor = torch.from_numpy(np.stack(encoded_msgs)).to(device)
        return encoded_tensor.float()

    def _extract_args(self, args, kwargs):
        if len(args) >= 2:
            imgs, msgs = args[0], args[1]
            rest = args[2:]
        else:
            imgs = kwargs.get("imgs", None)
            msgs = kwargs.get("msgs", None)
            rest = ()
            if imgs is None or msgs is None:
                raise ValueError(
                    "Expected forward(imgs, msgs, ...) but couldn't find imgs/msgs in args/kwargs."
                )

        return imgs, msgs, rest

    def _ecc_forward_pre_hook(self, module, args, kwargs):
        """
        Hook signature with with_kwargs=True:
          - args: tuple of positional args passed to forward
          - kwargs: dict of keyword args passed to forward
        Must return (new_args, new_kwargs).
        """
        imgs, msgs, rest = self._extract_args(args, kwargs)

        if msgs.shape[1] == self.message_bits and self.message_bits != self.total_bits:
            msgs = self._apply_ecc_encoding(msgs)
        elif msgs.shape[1] != self.total_bits:
            raise ValueError(
                f"Message size mismatch: expected {self.message_bits} bits "
                f"(for ECC encoding) or {self.total_bits} bits (raw), got {msgs.shape[1]}"
            )

        if len(args) >= 2:
            new_args = (imgs, msgs, *rest)
            return new_args, kwargs
        else:
            kwargs["imgs"] = imgs
            kwargs["msgs"] = msgs
            return args, kwargs
