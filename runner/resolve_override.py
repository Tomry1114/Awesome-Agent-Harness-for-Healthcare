"""Pin a host name to given addresses for this process: MH_RESOLVE="host:ip[,ip...]" (several hosts separated by ";").

Useful when a local resolver answers an API host with an address whose server presents a certificate for another
name, so every call fails TLS ("Hostname mismatch") while public resolvers return the right address. TLS still
verifies the certificate against the host name: only the address lookup is replaced, as curl --resolve does. Unset,
nothing changes."""
import os
import socket

_ORIG = socket.getaddrinfo


def _table(spec):
    out = {}
    for part in (spec or "").split(";"):
        host, _, ips = part.strip().partition(":")
        if host and ips:
            out[host.strip().lower()] = [ip.strip() for ip in ips.split(",") if ip.strip()]
    return out


def install(spec=None):
    """Route lookups of the listed hosts to their addresses. -> the table installed ({} when nothing to do)."""
    table = _table(spec if spec is not None else os.environ.get("MH_RESOLVE", ""))
    if not table:
        return {}

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        ips = table.get(str(host).lower()) if isinstance(host, str) else None
        if not ips:
            return _ORIG(host, port, family, type, proto, flags)
        out = []
        for ip in ips:
            out.extend(_ORIG(ip, port, family, type, proto, flags))
        return out
    socket.getaddrinfo = getaddrinfo
    return table
