"""Independent source/ELF metadata fixtures; no synthetic file is executed."""

from pathlib import Path
import hashlib
import struct

PROVIDER_SOURCES = (
    "adapters/ffi_provider.py",
    "adapters/skinny_moe.py",
    "scripts/hardware_canary.py",
)


def provider_sources(source: Path):
    hashes = {}
    for name in PROVIDER_SOURCES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ("# qualified test source: " + name + "\n").encode()
        path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def elf_metadata_fixture(payload: bytes):
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    # ELF64 LE ET_DYN x86-64 header, deliberately without loadable segments.
    # This is only a metadata test; actual preflight still invokes readelf and vf.
    return (
        struct.pack(
            "<16sHHIQQQIHHHHHH", ident, 3, 62, 1, 0, 0, 0, 0, 64, 56, 0, 64, 0, 0
        )
        + payload
    )
