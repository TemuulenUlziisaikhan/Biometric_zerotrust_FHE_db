from __future__ import annotations

API_VERSION = "v1"


class BiometricClientError(RuntimeError):
    pass


class TransportError(BiometricClientError):
    pass


class ServerError(BiometricClientError):
    pass
