"""Shared terminal/notebook progress display; no training state or randomness."""

from tqdm.auto import tqdm


def progress_bar(iterable=None, *, total=None, desc, unit, enabled=False,
                 position=0, leave=False):
    return tqdm(iterable, total=total, desc=desc, unit=unit,
                disable=not enabled, dynamic_ncols=True,
                position=position, leave=leave, mininterval=.2)


def progress_message(message, *, enabled=False):
    if enabled:
        tqdm.write(message)
    else:
        print(message, flush=True)
