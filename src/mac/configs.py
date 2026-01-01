import warnings
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self
from dataclasses import dataclass

# Research-only deterministic key (32 bytes = 256-bit) and password.
# Secret key must be hex-encoded: 64 hex chars = 32 bytes.
RESEARCH_ONLY_SECRET_KEY_HEX = "0123456789abcdef" * 4
RESEARCH_ONLY_PASSWORD = "change this password"

DEFAULT_SECRET_KEY_BYTES = bytes.fromhex(RESEARCH_ONLY_SECRET_KEY_HEX)
DEFAULT_PASSWORD_BYTES = bytes(RESEARCH_ONLY_PASSWORD, "UTF-8")


class MACSettings(BaseSettings):
    """
    verification notes:
        MAC-over-message-bit verification fails frequently because watermark extraction introduces bit errors.
        We use a shared password for robust verification in experiments.
    """

    model_config = SettingsConfigDict(
        env_file=str((Path(__file__).resolve().parent / "../../.env").resolve()),
        env_ignore_empty=True,
        extra="ignore",
    )

    MAC_SECRET_KEY: bytes = DEFAULT_SECRET_KEY_BYTES
    PASSWORD_BYTES: bytes = DEFAULT_PASSWORD_BYTES
    ENVIRONMENT: Literal["local", "production"] = "local"
    MIN_BYTES: int = 32

    def _warn_or_error(self, message: str) -> None:
        if self.ENVIRONMENT == "local":
            warnings.warn(message, stacklevel=2)
        else:
            raise ValueError(message)

    def _check_default_mac_key(self) -> None:
        if self.MAC_SECRET_KEY == DEFAULT_SECRET_KEY_BYTES:
            self._warn_or_error(
                "MAC_SECRET_KEY is set to the research default key. "
                "This is intended for reproducibility only and MUST NOT be used in production.\n"
                f'Generate a new key with: python -c "import secrets; print(secrets.token_hex({self.MIN_BYTES}))"'
            )

    def _check_default_password(self) -> None:
        if self.PASSWORD_BYTES == DEFAULT_PASSWORD_BYTES:
            self._warn_or_error(
                "PASSWORD is set to the research default password. "
                "This is intended for reproducibility only and MUST NOT be used in production.\n"
                'Generate a new password with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )

    def _check_mac_key_length(self) -> None:
        if len(self.MAC_SECRET_KEY) < self.MIN_BYTES:
            raise ValueError(
                f"MAC_SECRET_KEY too short: {len(self.MAC_SECRET_KEY)} bytes; "
                f"minimum {self.MIN_BYTES} bytes required."
            )

    def _check_password_nonempty(self) -> None:
        if len(self.PASSWORD_BYTES) == 0:
            raise ValueError("PASSWORD must not be empty.")

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        self._check_mac_key_length()
        self._check_password_nonempty()
        self._check_default_mac_key()
        self._check_default_password()
        return self


mac_settings = MACSettings()


@dataclass(frozen=True, slots=True)
class DetectionProbabilityScaling:
    """Configuration for combining model and MAC detection probabilities.

    Formula: combined_prob = scaling_model * model_prob + scaling_mac * mac_prob
    where mac_prob = matching_tag_bits / total_tag_bits

    - VideoSeal 1.0 provides a model probability, so the default weighting is:

      scaling_model = 0.5
      scaling_mac   = 0.5

    - StegaStamp and REVMark do not provide a model probability. In these cases, it is
    preferable to rely entirely on MAC matching:

        scaling_model = 0.0
        scaling_mac   = 1.0
    """

    scaling_model: float = 0.5  # Weight for model's probability
    scaling_mac: float = 0.5  # Weight for MAC verification probability
