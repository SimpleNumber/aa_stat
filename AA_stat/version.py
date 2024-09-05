try:
    from pyteomics.version import VersionInfo
except ImportError:
    from pyteomics.version import _VersionInfo as VersionInfo

__version__ = '2.6a1'

version_info = VersionInfo(__version__)
version = __version__
