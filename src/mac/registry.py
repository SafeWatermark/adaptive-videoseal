from mac.hmac_provider import HMACProvider

mac_provider_registry = {
    "hmac": HMACProvider,
    "default": HMACProvider,
}
