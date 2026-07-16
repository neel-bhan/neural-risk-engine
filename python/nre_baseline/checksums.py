"""Checksums shared by the M6 generator and M7 loader."""

from pathlib import Path


def fnv1a64_bytes(content: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in content:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def fnv1a64_file(path: Path) -> str:
    return fnv1a64_bytes(path.read_bytes())

