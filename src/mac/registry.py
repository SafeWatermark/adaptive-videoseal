from mac.hmac_provider import HMACProvider

mac_provider_registry = {
    "hmac_sha256": HMACProvider,
    "default": HMACProvider,
}
