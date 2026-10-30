"""Bounded host resource preflight; snapshots are not reservations or leases."""
import os
from dataclasses import dataclass
from pathlib import Path

class AdmissionError(RuntimeError):
    pass

def parse_meminfo(text):
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] in ('MemAvailable:', 'SwapFree:'):
            key = parts[0][:-1]
            if key in values or len(parts) != 3 or parts[2] != 'kB' or not parts[1].isascii() or not parts[1].isdigit():
                raise AdmissionError('invalid or duplicated memory metric')
            values[key] = int(parts[1]) * 1024
    if set(values) != {'MemAvailable', 'SwapFree'}:
        raise AdmissionError('missing memory metric')
    return values['MemAvailable'], values['SwapFree']

@dataclass(frozen=True)
class ResourcePolicy:
    min_memory: int
    min_swap: int
    min_disk: int
    projected_memory: int = 0

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.min_memory, self.min_swap, self.min_disk, self.projected_memory)):
            raise ValueError('resource limits must be nonnegative integers')

    def check_values(self, memory, swap, disk):
        if any(type(x) is not int or x < 0 for x in (memory, swap, disk)):
            raise AdmissionError('invalid resource snapshot')
        if memory < self.min_memory + self.projected_memory or swap < self.min_swap or disk < self.min_disk:
            raise AdmissionError('resource floor or projected memory not satisfied')
        return {'mem_available': memory, 'swap_free': swap, 'disk_free': disk}

    def check(self, path):
        try:
            memory, swap = parse_meminfo(Path('/proc/meminfo').read_text(encoding='ascii'))
            fs = os.statvfs(path)
            return self.check_values(memory, swap, fs.f_bavail * fs.f_frsize)
        except OSError as exc:
            raise AdmissionError('resource snapshot unavailable') from exc
