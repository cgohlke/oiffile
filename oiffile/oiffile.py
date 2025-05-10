# oiffile.py

# Copyright (c) 2012-2025, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read Olympus image files (OIF and OIB).

Oiffile is a Python library to read image and metadata from Olympus Image
Format files. OIF is the native file format of the Olympus FluoView(tm)
software for confocal microscopy.

There are two variants of the format:

- OIF (Olympus Image File) is a multi-file format that includes a main setting
  file (.oif) and an associated directory with data and setting files (.tif,
  .bmp, .txt, .pty, .roi, and .lut).

- OIB (Olympus Image Binary) is a compound document file, storing OIF and
  associated files within a single file.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2025.5.10

Quickstart
----------

Install the oiffile package and all dependencies from the
`Python Package Index <https://pypi.org/project/oiffile/>`_::

    python -m pip install -U "oiffile[all]"

View image and metadata stored in an OIF or OIB file::

    python -m oiffile file.oif

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/oiffile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.10.11, 3.11.9, 3.12.10, 3.13.3 64-bit
- `NumPy <https://pypi.org/project/numpy/>`_ 2.2.5
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2025.5.10

Revisions
---------

2025.5.10

- Remove doctest command line option.
- Support Python 3.14.

2025.1.1

- Improve type hints.
- Drop support for Python 3.9, support Python 3.13.

2024.5.24

- Support NumPy 2.
- Fix docstring examples not correctly rendered on GitHub.

2023.8.30

- Fix linting issues.
- Add py.typed marker.
- Drop support for Python 3.8 and numpy < 1.22 (NEP29).

2022.9.29

- Switch to Google style docstrings.

2022.2.2

- Add type hints.
- Add main function.
- Add FileSystemAbc abstract base class.
- Remove OifFile.tiffs (breaking).
- Drop support for Python 3.7 and numpy < 1.19 (NEP29).

2021.6.6

- Fix unclosed file warnings.

2020.9.18

- Remove support for Python 3.6 (NEP 29).
- Support os.PathLike file names.
- Fix unclosed files.

2020.1.18

- Fix indentation error.

2020.1.1

- Support multiple image series.
- Parse shape and dtype from settings file.
- Remove support for Python 2.7 and 3.5.
- Update copyright.

Notes
-----

No specification document is available.

Tested only with files produced on Olympus FV1000 hardware.

Examples
--------

Read the image from an OIB file as numpy array:

>>> image = imread('test.oib')
>>> image.shape
(3, 256, 256)
>>> image[:, 95, 216]
array([820,  50, 436], dtype=uint16)

Read the image from a single TIFF file in an OIB file:

>>> from tifffile import natural_sorted
>>> with OifFile('test.oib') as oib:
...     filename = natural_sorted(oib.glob('*.tif'))[0]
...     image = oib.asarray(filename)
...
>>> filename
'Storage00001/s_C001.tif'
>>> print(image[95, 216])
820

Access metadata and the OIB main file:

>>> with OifFile('test.oib') as oib:
...     oib.axes
...     oib.shape
...     oib.dtype
...     dataname = oib.mainfile['File Info']['DataName']
...
'CYX'
(3, 256, 256)
dtype('uint16')
>>> dataname
'Cell 1 mitoEGFP.oib'

Extract the OIB file content to an OIF file and associated data directory:

>>> import tempfile
>>> tempdir = tempfile.mkdtemp()
>>> oib2oif('test.oib', location=tempdir)
Saving ... done.

Read the image from the extracted OIF file:

>>> image = imread(f'{tempdir}/{dataname[:-4]}.oif')
>>> image[:, 95, 216]
array([820,  50, 436], dtype=uint16)

Read OLE compound file and access the 'OibInfo.txt' settings file:

>>> with CompoundFile('test.oib') as com:
...     info = com.open_file('OibInfo.txt')
...     len(com.files())
...
14
>>> info = SettingsFile(info, 'OibInfo.txt')
>>> info['OibSaveInfo']['Version']
'2.0.0.0'

"""

from __future__ import annotations

__version__ = '2025.5.10'

__all__ = [
    '__version__',
    'imread',
    'oib2oif',
    'OifFile',
    'OifFileError',
    'OibFileSystem',
    'OifFileSystem',
    'FileSystemAbc',
    'SettingsFile',
    'CompoundFile',
    'filetime',
]

import abc
import os
import re
import struct
import sys
from datetime import datetime, timezone
from glob import glob
from io import BytesIO
from typing import TYPE_CHECKING

import numpy
from tifffile import TiffFile, TiffSequence, natural_sorted

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator
    from typing import IO, Any, BinaryIO, Literal

    from numpy.typing import NDArray


def imread(filename: str | os.PathLike[Any], /, **kwargs: Any) -> NDArray[Any]:
    """Return image data from OIF or OIB file.

    Parameters:
        filename:
            Path to OIB or OIF file.
        **kwargs:
            Additional arguments passed to :py:meth:`OifFile.asarray`.

    """
    with OifFile(filename) as oif:
        result = oif.asarray(**kwargs)
    return result


def oib2oif(
    filename: str | os.PathLike[Any],
    /,
    location: str = '',
    *,
    verbose: int = 1,
) -> None:
    """Convert OIB file to OIF.

    Parameters:
        filename:
            Name of OIB file to convert.
        location:
            Directory, where files are written.
        verbose:
            Level of printed status messages.

    """
    with OibFileSystem(filename) as oib:
        oib.saveas_oif(location=location, verbose=verbose)


class OifFileError(Exception):
    """Exception to raise issues with OIF or OIB structure."""


class OifFile:
    """Olympus Image File.

    Parameters:
        filename: Path to OIB or OIF file.

    """

    filename: str
    """Name of OIB or OIF file."""

    filesystem: FileSystemAbc
    """Underlying file system instance."""

    mainfile: SettingsFile
    """Main settings."""

    _files_flat: dict[str, str]
    _series: tuple[TiffSequence, ...] | None

    def __init__(self, filename: str | os.PathLike[Any], /) -> None:
        self.filename = filename = os.fspath(filename)
        if filename.lower().endswith('.oif'):
            self.filesystem = OifFileSystem(filename)
        else:
            self.filesystem = OibFileSystem(filename)
        self.mainfile = self.filesystem.settings
        # map file names to storage names (flattened name space)
        self._files_flat = {
            os.path.basename(f): f for f in self.filesystem.files()
        }
        self._series = None

    def open_file(self, filename: str, /) -> BinaryIO:
        """Return open file object from path name.

        Parameters:
            filename: Name of file to open.

        """
        try:
            return self.filesystem.open_file(
                self._files_flat.get(filename, filename)
            )
        except (KeyError, OSError) as exc:
            raise FileNotFoundError(f'No such file: {filename}') from exc

    def glob(self, pattern: str = '*', /) -> Iterator[str]:
        """Return iterator over unsorted file names matching pattern.

        Parameters:
            pattern: File glob pattern.

        """
        if pattern == '*':
            return self.filesystem.files()
        ptrn = re.compile(pattern.replace('.', r'\.').replace('*', '.*'))
        return (f for f in self.filesystem.files() if ptrn.match(f))

    @property
    def is_oib(self) -> bool:
        """File has OIB format."""
        return isinstance(self.filesystem, OibFileSystem)

    @property
    def axes(self) -> str:
        """Character codes for dimensions in image array according to mainfile.

        This might differ from the axes order of series.

        """
        return str(self.mainfile['Axis Parameter Common']['AxisOrder'][::-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of image data according to mainfile.

        This might differ from the shape of series.

        """
        size = {
            self.mainfile[f'Axis {i} Parameters Common']['AxisCode']: int(
                self.mainfile[f'Axis {i} Parameters Common']['MaxSize']
            )
            for i in range(8)
        }
        return tuple(size[ax] for ax in self.axes)

    @property
    def dtype(self) -> numpy.dtype[Any]:
        """Type of image data according to mainfile.

        This might differ from the dtype of series.

        """
        bitcount = int(
            self.mainfile['Reference Image Parameter']['ValidBitCounts']
        )
        return numpy.dtype('<u2' if bitcount > 8 else '<u2')

    @property
    def series(self) -> tuple[TiffSequence, ...]:
        """Sequence of series of TIFF files with matching names."""
        if self._series is not None:
            return self._series
        tiffiles: dict[str, list[str]] = {}
        for fname in self.glob('*.tif'):
            key = ''.join(
                c for c in os.path.split(fname)[-1][:-4] if c.isalpha()
            )
            if key in tiffiles:
                tiffiles[key].append(fname)
            else:
                tiffiles[key] = [fname]
        series = tuple(
            TiffSequence(
                natural_sorted(files), imread=self.asarray, pattern='axes'
            )
            for files in tiffiles.values()
        )
        if len(series) > 1:
            series = tuple(reversed(sorted(series, key=lambda x: len(x))))
        self._series = series
        return series

    def asarray(self, series: int | str = 0, **kwargs: Any) -> NDArray[Any]:
        """Return image data from TIFF file(s) as numpy array.

        Parameters:
            series:
                Specifies which series to return as array.
            kwargs:
                Additional parameters passed to :py:meth:`TiffFile.asarray`
                or :py:meth:`TiffSequence.asarray`.

        """
        if isinstance(series, int):
            return self.series[series].asarray(**kwargs)
        fh = self.open_file(series)
        try:
            with TiffFile(fh, name=series) as tif:
                result = tif.asarray(**kwargs)
        finally:
            fh.close()
        return result

    def close(self) -> None:
        """Close file handle."""
        self.filesystem.close()

    def __enter__(self) -> OifFile:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        filename = os.path.split(self.filename)[-1]
        return f'<{self.__class__.__name__} {filename!r}>'

    def __str__(self) -> str:
        # info = self.mainfile['Version Info']
        return indent(
            repr(self),
            f'axes: {self.axes}',
            f'shape: {self.shape}',
            f'dtype: {self.dtype}',
            # f'system name: {info.get("SystemName", "None")}',
            # f'system version: {info.get("SystemVersion", "None")}',
            # f'file version: {info.get("FileVersion", "None")}',
            # indent(f'series: {len(self.series)}', *self.series),
            f'series: {len(self.series)}',
        )


class FileSystemAbc(metaclass=abc.ABCMeta):
    """Abstract base class for structures with key."""

    filename: str
    """Name of OIB or OIF file."""

    name: str
    """Name from settings file."""

    version: str
    """Version from settings file."""

    mainfile: str
    """Name of main settings file."""

    settings: SettingsFile
    """Main settings."""

    @abc.abstractmethod
    def open_file(self, filename: str, /) -> BinaryIO:
        """Return file object from path name.

        Parameters:
            filename: Name of file to open.

        """

    @abc.abstractmethod
    def files(self) -> Iterator[str]:
        """Return iterator over unsorted files in FileSystem."""

    def close(self) -> None:
        """Close file handle."""

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__} {os.path.split(self.filename)[-1]!r}>'
        )


class OifFileSystem(FileSystemAbc):
    """Olympus Image File file system.

    Parameters:
        filename:
            Name of OIF file.
        storage_ext:
            Name extension of storage directory.

    """

    filename: str
    name: str
    version: str
    mainfile: str
    settings: SettingsFile
    _files: list[str]
    _path: str

    def __init__(
        self, filename: str | os.PathLike[Any], /, storage_ext: str = '.files'
    ):
        self.filename = filename = os.fspath(filename)
        self._path, self.mainfile = os.path.split(os.path.abspath(filename))
        self.settings = SettingsFile(filename, name=self.mainfile)
        self.name = self.settings['ProfileSaveInfo']['Name']
        self.version = self.settings['ProfileSaveInfo']['Version']
        # check that storage directory exists
        storage = os.path.join(self._path, self.mainfile + storage_ext)
        if not os.path.exists(storage) or not os.path.isdir(storage):
            raise OSError(
                f'OIF storage path not found: {self.mainfile}{storage_ext}'
            )
        # list all files
        pathlen = len(self._path + os.path.sep)
        self._files = [self.mainfile]
        for f in glob(os.path.join(storage, '*')):
            self._files.append(f[pathlen:])

    def open_file(self, filename: str, /) -> BinaryIO:
        """Return file object from path name.

        The returned file object must be closed by the user.

        Parameters:
            filename: Name of file to open.

        """
        return open(os.path.join(self._path, filename), 'rb')

    def files(self) -> Iterator[str]:
        """Return iterator over unsorted files in OIF."""
        return iter(self._files)

    def close(self) -> None:
        """Close file handle."""

    def __enter__(self) -> OifFileSystem:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __str__(self) -> str:
        return indent(
            repr(self),
            f'name: {self.name}',
            f'version: {self.version}',
            f'mainfile: {self.mainfile}',
        )


class OibFileSystem(FileSystemAbc):
    """Olympus Image Binary file system.

    Parameters:
        filename: Name of OIB file.

    """

    filename: str
    name: str
    version: str
    mainfile: str
    settings: SettingsFile
    com: CompoundFile
    compression: str
    _files: dict[str, str]
    _folders: dict[str, str]

    def __init__(self, filename: str | os.PathLike[Any], /) -> None:
        # open compound document and read OibInfo.txt settings
        self.filename = filename = os.fspath(filename)
        self.com = CompoundFile(filename)
        info = SettingsFile(self.com.open_file('OibInfo.txt'), 'OibInfo.txt')[
            'OibSaveInfo'
        ]
        self.name = info.get('Name', None)
        self.version = info.get('Version', None)
        self.compression = info.get('Compression', None)
        self.mainfile = info[info['MainFileName']]
        # map OIB file names to CompoundFile file names
        oibfiles = {os.path.split(i)[-1]: i for i in self.com.files()}
        self._files = {
            v: oibfiles[k] for k, v in info.items() if k.startswith('Stream')
        }
        # map storage names to directory names
        self._folders = {
            i[0]: i[1] for i in info.items() if i[0].startswith('Storage')
        }
        # read main settings file
        self.settings = SettingsFile(
            self.open_file(self.mainfile), name=self.mainfile
        )

    def open_file(self, filename: str, /) -> BinaryIO:
        """Return file object from case sensitive path name.

        Parameters:
            filename: Name of file to open.

        """
        try:
            return self.com.open_file(self._files[filename])
        except KeyError as exc:
            raise FileNotFoundError(f'No such file: {filename}') from exc

    def files(self) -> Iterator[str]:
        """Return iterator over unsorted files in OIB."""
        return iter(self._files.keys())

    def saveas_oif(self, location: str = '', *, verbose: int = 0) -> None:
        """Save all streams in OIB file as separate files.

        Raise OSError if target files or directories already exist.

        The main .oif file name and storage names are determined from the
        OibInfo.txt settings.

        Parameters:
            location:
                Directory, where files are written.
            verbose:
                Level of printed status messages.

        """
        if location and not os.path.exists(location):
            os.makedirs(location)
        mainfile = os.path.join(location, self.mainfile)
        if os.path.exists(mainfile):
            raise FileExistsError(mainfile + ' already exists')
        for folder in self._folders.keys():
            folder = os.path.join(location, self._folders.get(folder, ''))
            if os.path.exists(folder):
                raise FileExistsError(folder + ' already exists')
            os.makedirs(folder)
        if verbose:
            print('Saving', mainfile, end=' ')
        for f in self._files.keys():
            folder, name = os.path.split(f)
            folder = os.path.join(location, self._folders.get(folder, ''))
            path = os.path.join(folder, name)
            if verbose == 1:
                print(end='.')
            elif verbose > 1:
                print(path)
            with open(path, 'w+b') as fh:
                fh.write(self.open_file(f).read())
        if verbose == 1:
            print(' done.')

    def close(self) -> None:
        """Close file handle."""
        self.com.close()

    def __enter__(self) -> OibFileSystem:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __str__(self) -> str:
        return indent(
            repr(self),
            f'name: {self.name}',
            f'version: {self.version}',
            f'mainfile: {self.mainfile}',
            f'compression: {self.compression}',
        )


class SettingsFile(dict):  # type: ignore[type-arg]
    """Olympus settings file (oif, txt, pty, roi, lut).

    Settings files contain little endian utf-16 encoded strings, except for
    [ColorLUTData] sections, which contain uint8 binary arrays.

    Settings can be accessed as a nested dictionary {section: {key: value}},
    except for {'ColorLUTData': numpy array}.

    Parameters:
        file:
            Name of file or open file containing little endian UTF-16 string.
            File objects are closed.
        name:
            Human readable label of stream.

    """

    name: str
    """Name of settings."""

    def __init__(
        self,
        file: str | os.PathLike[Any] | IO[bytes],
        /,
        name: str | None = None,
    ) -> None:
        # read settings file and parse into nested dictionaries
        fh: IO[bytes]
        content: bytes
        content_list: list[bytes]

        dict.__init__(self)
        if isinstance(file, (str, os.PathLike)):
            self.name = os.path.split(file)[-1]
            fh = open(file, 'rb')
        else:
            self.name = str(name)
            fh = file

        try:
            content = fh.read()
        finally:
            fh.close()

        if content[:4] == b'\xff\xfe\x5b\x00':
            # UTF16 BOM
            content_list = content.rsplit(
                b'[\x00C\x00o\x00l\x00o\x00r\x00L\x00U\x00T\x00'
                b'D\x00a\x00t\x00a\x00]\x00\x0d\x00\x0a\x00',
                1,
            )
            if len(content_list) > 1:
                self['ColorLUTData'] = numpy.fromstring(
                    content_list[1],
                    dtype=numpy.uint8,  # type: ignore[call-overload]
                ).reshape(-1, 4)
            contents = content_list[0].decode('utf-16')
        elif content[:1] == b'[':
            # try UTF-8
            content_list = content.rsplit(b'[ColorLUTData]\r\n', 1)
            if len(content_list) > 1:
                self['ColorLUTData'] = numpy.fromstring(
                    content_list[1],
                    dtype=numpy.uint8,  # type: ignore[call-overload]
                ).reshape(-1, 4)
            try:
                contents = content_list[0].decode()
            except Exception as exc:
                raise ValueError('not a valid settings file') from exc
        else:
            raise ValueError('not a valid settings file')

        for line in contents.splitlines():
            line = line.strip()
            if line.startswith(';'):
                continue
            if line.startswith('[') and line.endswith(']'):
                self[line[1:-1]] = properties = {}
            else:
                key, value = line.split('=')
                properties[key] = astype(value)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name!r}>'

    def __str__(self) -> str:
        return indent(repr(self), format_dict(self))


class CompoundFile:
    """Compound Document File.

    A partial implementation of the "[MS-CFB] - v20120705, Compound File
    Binary File Format" specification by Microsoft Corporation.

    This should be able to read Olympus OIB and Zeiss ZVI files.

    Parameters:
        filename: Path to compound document file.

    """

    filename: str
    clsid: bytes | None
    version_minor: int
    version_major: int
    byteorder: Literal['<']
    dir_len: int
    fat_len: int
    dir_start: int
    mini_stream_cutof_size: int
    minifat_start: int
    minifat_len: int
    difat_start: int
    difat_len: int
    sec_size: int
    short_sec_size: int
    _files: dict[str, DirectoryEntry]
    _fat: list[Any]
    _minifat: list[Any]
    _difat: list[Any]
    _dirs: list[DirectoryEntry]

    MAXREGSECT = 0xFFFFFFFA
    DIFSECT = 0xFFFFFFFC
    FATSECT = 0xFFFFFFFD
    ENDOFCHAIN = 0xFFFFFFFE
    FREESECT = 0xFFFFFFFF
    MAXREGSID = 0xFFFFFFFA
    NOSTREAM = 0xFFFFFFFF

    def __init__(self, filename: str | os.PathLike[Any], /) -> None:
        self.filename = filename = os.fspath(filename)
        self._fh = open(filename, 'rb')
        try:
            self._fromfile()
        except Exception:
            self._fh.close()
            raise

    def _fromfile(self) -> None:
        """Initialize instance from file."""
        if self._fh.read(8) != b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            raise ValueError('not a compound document file')
        (
            self.clsid,
            self.version_minor,
            self.version_major,
            byteorder,
            sector_shift,
            mini_sector_shift,
            _,
            _,
            self.dir_len,
            self.fat_len,
            self.dir_start,
            _,
            self.mini_stream_cutof_size,
            self.minifat_start,
            self.minifat_len,
            self.difat_start,
            self.difat_len,
        ) = struct.unpack('<16sHHHHHHIIIIIIIIII', self._fh.read(68))

        if byteorder == 0xFFFE:
            self.byteorder = '<'
        else:
            # 0xFEFF
            # self.byteorder = '>'
            raise NotImplementedError('big-endian byte order not supported')

        if self.clsid == b'\x00' * 16:
            self.clsid = None
        if self.clsid is not None:
            raise OifFileError(f'cannot handle {self.clsid=!r}')

        if self.version_minor != 0x3E:
            raise OifFileError(f'cannot handle {self.version_minor=}')
        if mini_sector_shift != 0x0006:
            raise OifFileError(f'cannot handle {mini_sector_shift=}')
        if not (
            (self.version_major == 0x4 and sector_shift == 0x000C)
            or (
                self.version_major == 0x3
                and sector_shift == 0x0009
                and self.dir_len == 0
            )
        ):
            raise OifFileError(
                f'cannot handle {self.version_major=} and {sector_shift=}'
            )

        self.sec_size = 2**sector_shift
        self.short_sec_size = 2**mini_sector_shift

        secfmt = '<' + ('I' * (self.sec_size // 4))
        # read DIFAT
        self._difat = list(
            struct.unpack('<' + ('I' * 109), self._fh.read(436))
        )
        nextsec = self.difat_start
        for i in range(self.difat_len):
            if nextsec >= CompoundFile.MAXREGSID:
                raise OifFileError(f'{nextsec=} >= {CompoundFile.MAXREGSID=}')
            sec = struct.unpack(secfmt, self._sec_read(nextsec))
            self._difat.extend(sec[:-1])
            nextsec = sec[-1]
        # if nextsec != CompoundFile.ENDOFCHAIN: raise OifFileError()
        self._difat = self._difat[: self.fat_len]
        # read FAT
        self._fat = []
        for secid in self._difat:
            self._fat.extend(struct.unpack(secfmt, self._sec_read(secid)))
        # read mini FAT
        self._minifat = []
        for i, sector in enumerate(self._sec_chain(self.minifat_start)):
            if i >= self.minifat_len:
                break
            self._minifat.extend(struct.unpack(secfmt, sector))
        # read directories
        self._dirs = []
        for sector in self._sec_chain(self.dir_start):
            for i in range(0, self.sec_size, 128):
                direntry = DirectoryEntry(
                    sector[i : i + 128], self.version_major
                )
                self._dirs.append(direntry)
        # read root storage
        if len(self._dirs) <= 0:
            raise OifFileError('no directories found')
        root = self._dirs[0]
        if root.name != 'Root Entry':
            raise OifFileError(f'no root directory found, got {root.name!r}')
        if root.create_time is not None:  # and root.modify_time is None
            raise OifFileError(f'invalid {root.create_time=}')
        if root.stream_size % self.short_sec_size != 0:
            raise OifFileError(
                f'{root.stream_size=} does not match {self.short_sec_size=}'
            )
        # read mini stream
        self._ministream = b''.join(self._sec_chain(root.sector_start))
        self._ministream = self._ministream[: root.stream_size]
        # map stream/file names to directory entries
        nostream = CompoundFile.NOSTREAM
        join = '/'.join  # os.path.sep.join
        dirs = self._dirs
        visited = [False] * len(self._dirs)

        def parse(
            dirid: int, path: list[str]
        ) -> Generator[tuple[str, DirectoryEntry]]:
            # return iterator over all file names and their directory entries
            # TODO: replace with red-black tree parser
            if dirid == nostream or visited[dirid]:
                return
            visited[dirid] = True
            de = dirs[dirid]
            if de.is_stream:
                yield join(path + [de.name]), de
            yield from parse(de.left_sibling_id, path)
            yield from parse(de.right_sibling_id, path)
            if de.is_storage:
                yield from parse(de.child_id, path + [de.name])

        self._files = dict(parse(self._dirs[0].child_id, []))

    def _read_stream(self, direntry: DirectoryEntry, /) -> bytes:
        """Return content of stream."""
        if direntry.stream_size < self.mini_stream_cutof_size:
            result = b''.join(self._mini_sec_chain(direntry.sector_start))
        else:
            result = b''.join(self._sec_chain(direntry.sector_start))
        return result[: direntry.stream_size]

    def _sec_read(self, secid: int, /) -> bytes:
        """Return content of sector from file."""
        self._fh.seek(self.sec_size + secid * self.sec_size)
        return self._fh.read(self.sec_size)

    def _sec_chain(self, secid: int, /) -> Generator[bytes]:
        """Return iterator over FAT sector chain content."""
        while secid != CompoundFile.ENDOFCHAIN:
            if secid <= CompoundFile.MAXREGSECT:
                yield self._sec_read(secid)
            secid = self._fat[secid]

    def _mini_sec_read(self, secid: int, /) -> bytes:
        """Return content of sector from mini stream."""
        pos = secid * self.short_sec_size
        return self._ministream[pos : pos + self.short_sec_size]

    def _mini_sec_chain(self, secid: int, /) -> Generator[bytes]:
        """Return iterator over mini FAT sector chain content."""
        while secid != CompoundFile.ENDOFCHAIN:
            if secid <= CompoundFile.MAXREGSECT:
                yield self._mini_sec_read(secid)
            secid = self._minifat[secid]

    def files(self) -> Iterable[str]:
        """Return sequence of file names."""
        return self._files.keys()

    def direntry(self, filename: str, /) -> DirectoryEntry:
        """Return DirectoryEntry of filename.

        Parameters:
            filename: Name of file.

        """
        return self._files[filename]

    def open_file(self, filename: str, /) -> BytesIO:
        """Return stream as file like object.

        Parameters:
            filename: Name of file to open.

        """
        return BytesIO(self._read_stream(self._files[filename]))

    def format_tree(self) -> str:
        """Return formatted string with list of all files."""
        return '\n'.join(natural_sorted(self.files()))

    def close(self) -> None:
        """Close file handle."""
        self._fh.close()

    def __enter__(self) -> CompoundFile:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        filename = os.path.split(self.filename)[-1]
        return f'<{self.__class__.__name__} {filename!r}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            *(
                f'{attr}: {getattr(self, attr)}'
                for attr in (
                    'clsid',
                    'version_minor',
                    'version_major',
                    'byteorder',
                    'dir_len',
                    'fat_len',
                    'dir_start',
                    'mini_stream_cutof_size',
                    'minifat_start',
                    'minifat_len',
                    'difat_start',
                    'difat_len',
                )
            ),
        )


class DirectoryEntry:
    """Compound Document Directory Entry.

    Parameters:
        header:
            128 bytes compound document directory entry header.
        version_major:
            Major version of compound document.

    """

    __slots__ = (
        'name',
        'entry_type',
        'color',
        'left_sibling_id',
        'right_sibling_id',
        'child_id',
        'clsid',
        'user_flags',
        'create_time',
        'modify_time',
        'sector_start',
        'stream_size',
        'is_stream',
        'is_storage',
    )

    name: str
    entry_type: int
    color: int
    left_sibling_id: int
    right_sibling_id: int
    child_id: int
    clsid: bytes | None
    user_flags: int
    create_time: datetime | None
    modify_time: datetime | None
    sector_start: int
    stream_size: int
    is_stream: bool
    is_storage: bool

    def __init__(self, header: bytes, version_major: int, /) -> None:
        (
            name,
            name_len,
            self.entry_type,
            self.color,
            self.left_sibling_id,
            self.right_sibling_id,
            self.child_id,
            self.clsid,
            self.user_flags,
            create_time,
            modify_time,
            self.sector_start,
            self.stream_size,
        ) = struct.unpack('<64sHBBIII16sIQQIQ', header)

        if version_major == 3:
            self.stream_size = struct.unpack('<I', header[-8:-4])[0]
        if self.clsid == b'\000' * 16:
            self.clsid = None

        if name_len % 2 != 0 or name_len > 64:
            raise OifFileError(f'invalid {name_len=}')
        if self.color not in (0, 1):
            raise OifFileError(f'invalid {self.color=}')

        self.name = name[: name_len - 2].decode('utf-16')
        self.create_time = filetime(create_time)
        self.modify_time = filetime(modify_time)
        self.is_stream = self.entry_type == 2
        self.is_storage = self.entry_type == 1

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name!r}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            *(f'{attr}: {getattr(self, attr)}' for attr in self.__slots__[1:]),
        )


def indent(*args: Any) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


def format_dict(
    adict: dict[str, Any],
    prefix: str = ' ',
    indent: str = ' ',
    bullets: tuple[str, str] = ('', ''),
    excludes: tuple[str] = ('_',),
    linelen: int = 79,
    trim: int = 1,
) -> str:
    """Return pretty-print of nested dictionary."""
    result = []
    for k, v in sorted(adict.items(), key=lambda x: str(x[0]).lower()):
        if any(k.startswith(e) for e in excludes):
            continue
        if isinstance(v, dict):
            v = '\n' + format_dict(
                v, prefix=prefix + indent, excludes=excludes, trim=0
            )
            result.append(f'{prefix}{bullets[1]}{k}: {v}')
        else:
            result.append((f'{prefix}{bullets[0]}{k}: {v}')[:linelen].rstrip())
    if trim > 0:
        result[0] = result[0][trim:]
    return '\n'.join(result)


def astype(arg: str, types: Iterable[type] | None = None) -> Any:
    """Return argument as one of types if possible.

    Parameters:
        arg:
            String representation of value.
        types:
            Possible types of value. By default, int, float, and str.

    """
    if arg[0] in '\'"':
        return arg[1:-1]
    if types is None:
        types = int, float, str
    for typ in types:
        try:
            return typ(arg)
        except (ValueError, TypeError, UnicodeEncodeError):
            pass
    return arg


def filetime(ft: int, /) -> datetime | None:
    """Return Python datetime from Microsoft FILETIME number.

    Parameters:
        ft: Microsoft FILETIME number.

    """
    if not ft:
        return None
    sec, nsec = divmod(ft - 116444736000000000, 10000000)
    return datetime.fromtimestamp(sec, timezone.utc).replace(
        microsecond=nsec // 10
    )


def main(argv: list[str] | None = None) -> int:
    """Oiffile command line usage main function.

    ``python -m oiffile file_or_directory``

    """
    if argv is None:
        argv = sys.argv

    if len(argv) != 2:
        print('Usage: python -m oiffile file_or_directory')
        return 0

    from matplotlib import pyplot
    from tifffile import imshow

    with OifFile(sys.argv[1]) as oif:
        print(oif)
        print(oif.mainfile)
        for series in oif.series:
            # print(series)
            image = series.asarray()
            figure = pyplot.figure()
            imshow(image, figure=figure)
        pyplot.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
