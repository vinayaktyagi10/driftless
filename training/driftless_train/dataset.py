"""IO-VNBD loader.

Expects the raw IO-VNBD trip files under training/data/io-vnbd/ (gitignored —
download separately per github.com/onyekpeu/IO-VNBD). Produces fixed-length
windows of (accel, gyro, mag) paired with ground-truth speed/position for
supervised training of the velocity model.
"""


class IoVnbdWindowedDataset:
    # TODO: torch.utils.data.Dataset — load trip CSVs, slice into
    # overlapping windows, align IMU rows to ground-truth speed/GPS rows.
    pass
