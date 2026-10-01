"""
Shared utilities for the greenhouse tomato multi-objective optimization project.

This preview module contains the process-level helpers used by every entry point:

- ``check_env_dll_path``  : fail-fast conda DLL search-path self-check (Windows).
- ``install_tiff_savefig``: process-wide redirection of every matplotlib ``savefig``
  to 600-dpi lossless TIFF, so no raster PNG side products are ever produced.
- ``setup_logging``       : dual-channel (console + file) logging bootstrap.
"""

import logging
import os
import sys
from pathlib import Path

_tiff_savefig_installed = False


def check_env_dll_path():
    """Fail fast when the active conda environment's ``Library/bin`` is missing from PATH.

    Without that directory on the DLL search path, plotting libraries crash silently on
    Windows (KERNELBASE.dll exception 0xc06d007f, no traceback) when a C extension is
    lazily loaded later in the run. The environment is derived from ``sys.executable``,
    so the check works for any conda installation without hard-coded paths.
    """
    exe_dir = Path(sys.executable).resolve().parent
    lib_bin = exe_dir / 'Library' / 'bin'
    if lib_bin.is_dir() and str(lib_bin).lower() not in os.environ.get('PATH', '').lower():
        sys.exit(f"[FATAL] conda env '{exe_dir.name}' Library\\bin is not on PATH; "
                 f"plotting C extensions would crash silently.\n"
                 f"Activate the environment first: conda activate {exe_dir.name}")


def install_tiff_savefig(dpi: int = 600):
    """Redirect every matplotlib ``savefig`` in this process to 600-dpi lossless TIFF.

    All call sites (``.png``/``.tif``/extension-less paths) are funneled to a single
    deflate-compressed TIFF named after the requested stem, with ``bbox_inches='tight'``
    enforced by default so legends, tick labels, and annotations are never clipped.
    Idempotent: calling twice is a no-op.
    """
    global _tiff_savefig_installed
    if _tiff_savefig_installed:
        return
    import io
    import matplotlib.pyplot as plt
    from PIL import Image

    def _savefig_with_tiff(*args, **kwargs):
        path = str(args[0]) if args else str(kwargs.get('fname', 'figure'))
        kwargs['dpi'] = dpi
        kwargs['bbox_inches'] = kwargs.get('bbox_inches', 'tight')
        tif_path = os.path.splitext(path)[0] + '.tif'
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches=kwargs['bbox_inches'])
        buf.seek(0)
        with Image.open(buf) as im:
            im.save(tif_path, format='TIFF', compression='tiff_deflate', dpi=(dpi, dpi))
        return None

    plt.savefig = _savefig_with_tiff
    _tiff_savefig_installed = True


def setup_logging(log_file: str = 'app.log', level=logging.INFO):
    """Bootstrap dual-channel logging (console + UTF-8 file) and return the logger.

    The file handler appends, so repeated runs accumulate a persistent operational
    record; the console handler mirrors every record for interactive use.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
