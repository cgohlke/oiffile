# oiffile.py

# Copyright (c) 2012-2020, Christoph Gohlke
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

"""Read Olympus(r) image files (OIF and OIB).

Oiffile is a Python library to read image and metadata from Olympus Image
Format files. OIF is the native file format of the Olympus FluoView(tm)
software for confocal microscopy.

There are two variants of the format:

* OIF (Olympus Image File) is a multi-file format that includes a main setting
  file (.oif) and an associated directory with data and setting files (.tif,
  .bmp, .txt, .pyt, .roi, and .lut).

* OIB (Olympus Image Binary) is a compound document file, storing OIF and
  associated files within a single file.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.9.18

Requirements
------------
* `CPython >= 3.7 <https://www.python.org>`_
* `Numpy 1.15 <https://www.numpy.org>`_
* `Tifffile 2020.6.3 <https://pypi.org/project/tifffile/>`_

Revisions
---------
2020.9.18
    Remove support for Python 3.6 (NEP 29).
    Support os.PathLike file names.
    Fix unclosed files.
2020.1.18
    Fix indentation error.
2020.1.1
    Support multiple image series.
    Parse shape and dtype from settings file.
    Remove support for Python 2.7 and 3.5.
    Update copyright.

Notes
-----
The API is not stable yet and might change between revisions.

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

>>> with OifFile('test.oib') as oib:
...     filename = natural_sorted(oib.glob('*.tif'))[0]
...     image = oib.asarray(filename)
>>> filename
'Storage00001/s_C001.tif'
>>> image[95, 216]
820

Access metadata and the OIB main file:

>>> with OifFile('test.oib') as oib:
...     oib.axes
...     oib.shape
...     oib.dtype
...     dataname = oib.mainfile['File Info']['DataName']
'CYX'
(3, 256, 256)
dtype('uint16')
>>> dataname
'Cell 1 mitoEGFP.oib'

Extract the OIB file content to an OIF file and associated data directory:

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
14
>>> info = SettingsFile(info, 'OibInfo.txt')
>>> info['OibSaveInfo']['Version']
'2.0.0.0'

"""

__version__ = '2020.9.18'

__all__ = (
    'imread',
    'oib2oif',
    'OifFile',
    'OifFileError',
    'OibFileSystem',
    'OifFileSystem',
    'SettingsFile',
    'CompoundFile',
    'filetime',
)

import os
import re
import struct
from io import BytesIO
from glob import glob
from datetime import datetime

import numpy

from tifffile import TiffFile, TiffSequence, lazyattr, natural_sorted


def imread(filename, *args, **kwargs):
    """Return image data from OIF or OIB file as numpy array.

    'args' and 'kwargs' are arguments to OifFile.asarray().

    """
    with OifFile(filename) as oif:
        result = oif.asarray(*args, **kwargs)
    return result


def oib2oif(filename, location='', verbose=1):
    """Convert OIB file to OIF."""
    with OibFileSystem(filename) as oib:
        oib.saveas_oif(location=location, verbose=verbose)


class OifFileError(Exception):
    """Exception to raise issues with OIF or OIB structure."""


class OifFile:
    """Olympus Image File.

    Attributes
    ----------
    mainfile : SettingsFile
        The main OIF settings.
    filesystem : OibFileSystem or OifFileSystem
        The underlying file system instance.
    series : tuple of tifffile.TiffSequence
        Sequence of TIFF files. Includes shape, dtype, and axes information.
    is_oib : bool
        True if OIB file.

    """

    def __init__(self, filename):
        """Open OIF or OIB file and read main settings."""
        self.filename = filename = os.fspath(filename)
        if filename.lower().endswith('.oif'):
            self.filesystem = OifFileSystem(filename)
            self.is_oib = False
        else:
            self.filesystem = OibFileSystem(filename)
            self.is_oib = True
        self.mainfile = self.filesystem.settings
        # map file names to storage names (flattened name space)
        self._files_flat = {
            os.path.basename(f): f for f in self.filesystem.files()
        }

    def open_file(self, filename):
        """Return open file object from path name."""
        try:
            return self.filesystem.open_file(
                self._files_flat.get(filename, filename)
            )
        except (KeyError, OSError) as exc:
            raise FileNotFoundError(f'No such file: {filename}') from exc

    def glob(self, pattern='*'):
        """Return iterator over unsorted file names matching pattern."""
        if pattern == '*':
            return self.filesystem.files()
        pattern = pattern.replace('.', r'\.').replace('*', '.*')
        pattern = re.compile(pattern)
        return (f for f in self.filesystem.files() if pattern.match(f))

    @property
    def axes(self):
        """Return order of axes in image data from mainfile."""
        return self.mainfile['Axis Parameter Common']['AxisOrder'][::-1]

    @property
    def shape(self):
        """Return shape of image data from mainfile."""
        size = {
            self.mainfile[f'Axis {i} Parameters Common']['AxisCode']: int(
                self.mainfile[f'Axis {i} Parameters Common']['MaxSize']
            )
            for i in range(8)
        }
        return tuple(size[ax] for ax in self.axes)

    @property
    def dtype(self):
        """Return dtype of image data from mainfile."""
        bitcount = int(
            self.mainfile['Reference Image Parameter']['ValidBitCounts']
        )
        return numpy.dtype('<u2' if bitcount > 8 else '<u2')

    @lazyattr
    def series(self):
        """Return tuple of TiffSequence of sorted TIFF files."""
        series = {}
        for fname in self.glob('*.tif'):
            key = ''.join(
                c for c in os.path.split(fname)[-1][:-4] if c.isalpha()
            )
            if key in series:
                series[key].append(fname)
            else:
                series[key] = [fname]
        series = [
            TiffSequence(
                natural_sorted(files), imread=self.asarray, pattern='axes'
            )
            for files in series.values()
        ]
        if len(series) > 1:
            series = tuple(
                reversed(sorted(series, key=lambda x: len(x.files)))
            )
        return series

    @property
    def tiffs(self):
        """Return first TiffSequence."""
        # required for compatibility with cmapfile < 2020.1.1
        return self.series[0]

    def asarray(self, series=0, **kwargs):
        """Return image data from TIFF file(s) as numpy array.

        By default the data from the TIFF files in the first image series
        is returned.

        The kwargs parameters are passed to the asarray functions of the
        TiffFile or TiffSequence instances.

        """
        if isinstance(series, int):
            return self.series[series].asarray(**kwargs)
        try:
            fh = self.open_file(series)
            with TiffFile(fh, name=series) as tif:
                result = tif.asarray(**kwargs)
        finally:
            fh.close()
        return result

    def close(self):
        """Close file handle."""
        self.filesystem.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self):
        """Return string with information about OifFile."""
        # info = self.mainfile['Version Info']
        s = [
            self.__class__.__name__,
            os.path.normpath(os.path.normcase(self.filename)),
            f'axes: {self.axes}',
            'shape: {}'.format(', '.join(str(i) for i in self.shape)),
            f'dtype: {self.dtype}',
            # f'system name: {info.get("SystemName", "None")}',
            # f'system version: {info.get("SystemVersion", "None")}',
            # f'file version: {info.get("FileVersion", "None")}',
        ]
        if len(self.series) > 1:
            s.append(f'series: {len(self.series)}')
        return '\n '.join(s)


class OifFileSystem:
    """Olympus Image File file system."""

    def __init__(self, filename, storage_ext='.files'):
        """Open OIF file and read settings."""
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

    def open_file(self, filename):
        """Return file object from path name.

        File object must be closed.

        """
        return open(os.path.join(self._path, filename), 'rb')

    def files(self):
        """Return iterator over unsorted files in OIF."""
        return iter(self._files)

    def glob(self, pattern):
        """Return iterator of path names matching the specified pattern."""
        pattern = pattern.replace('.', r'\.').replace('*', '.*')
        pattern = re.compile(pattern)
        return (f for f in self.files() if pattern.match(f))

    def close(self):
        """Close file handle."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self):
        """Return string with information about OifFileSystem."""
        return '\n '.join(
            (
                self.__class__.__name__,
                os.path.normpath(os.path.normcase(self.filename)),
                f'name: {self.name}',
                f'version: {self.version}',
                f'mainfile: {self.mainfile}',
            )
        )


class OibFileSystem:
    """Olympus Image Binary file system."""

    def __init__(self, filename):
        """Open compound document and read OibInfo.txt settings."""
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

    def open_file(self, filename):
        """Return file object from case sensitive path name."""
        try:
            return self.com.open_file(self._files[filename])
        except KeyError as exc:
            raise FileNotFoundError(f'No such file: {filename}') from exc

    def files(self):
        """Return iterator over unsorted files in OIB."""
        return iter(self._files.keys())

    def saveas_oif(self, location='', verbose=0):
        """Save all streams in OIB file as separate files.

        Raise OSError if target files or directories already exist.

        The main .oif file name and storage names are determined from the
        OibInfo.txt settings.

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

    def close(self):
        """Close file handle."""
        self.com.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self):
        """Return string with information about OibFileSystem."""
        return '\n '.join(
            (
                self.__class__.__name__,
                os.path.normpath(os.path.normcase(self.filename)),
                f'name: {self.name}',
                f'version: {self.version}',
                f'mainfile: {self.mainfile}',
                f'compression: {self.compression}',
            )
        )


class SettingsFile(dict):
    """Olympus settings file (oif, txt, pyt, roi, lut).

    Settings files contain little endian utf-16 encoded strings, except for
    [ColorLUTData] sections, which contain uint8 binary arrays.

    Settings can be accessed as a nested dictionary {section: {key: value}},
    except for {'ColorLUTData': numpy array}.

    """

    def __init__(self, arg, name=None):
        """Read settings file and parse into nested dictionaries.

        Parameters
        ----------
        arg : str or file object
            Name of file or open file containing little endian UTF-16 string.
            File objects are closed by this function.
        name : str
            Human readable label of stream.

        """
        dict.__init__(self)
        if isinstance(arg, (str, os.PathLike)):
            self.name = os.fspath(arg)
            stream = open(arg, 'rb')
        else:
            self.name = str(name)
            stream = arg

        try:
            content = stream.read()
        finally:
            stream.close()

        if content[:4] == b'\xFF\xFE\x5B\x00':
            # UTF16 BOM
            content = content.rsplit(
                b'[\x00C\x00o\x00l\x00o\x00r\x00L\x00U\x00T\x00'
                b'D\x00a\x00t\x00a\x00]\x00\x0D\x00\x0A\x00',
                1,
            )
            if len(content) > 1:
                self['ColorLUTData'] = numpy.fromstring(
                    content[1], 'uint8'
                ).reshape(-1, 4)
            content = content[0].decode('utf-16')
        elif content[:1] == b'[':
            # try UTF-8
            content = content.rsplit(b'[ColorLUTData]\r\n', 1)
            if len(content) > 1:
                self['ColorLUTData'] = numpy.fromstring(
                    content[1], 'uint8'
                ).reshape(-1, 4)
            try:
                content = content[0].decode()
            except Exception as exc:
                raise ValueError('not a valid settings file') from exc
        else:
            raise ValueError('not a valid settings file')

        for line in content.splitlines():
            line = line.strip()
            if line.startswith(';'):
                continue
            if line.startswith('[') and line.endswith(']'):
                self[line[1:-1]] = properties = {}
            else:
                key, value = line.split('=')
                properties[key] = astype(value)

    def __str__(self):
        """Return string with information about SettingsFile."""
        return '\n '.join((self.__class__.__name__, format_dict(self)))


class CompoundFile:
    """Compound Document File.

    A partial implementation of the "[MS-CFB] - v20120705, Compound File
    Binary File Format" specification by Microsoft Corporation.

    This should be able to read Olympus OIB and Zeiss ZVI files.

    """

    MAXREGSECT = 0xFFFFFFFA
    DIFSECT = 0xFFFFFFFC
    FATSECT = 0xFFFFFFFD
    ENDOFCHAIN = 0xFFFFFFFE
    FREESECT = 0xFFFFFFFF
    MAXREGSID = 0xFFFFFFFA
    NOSTREAM = 0xFFFFFFFF

    def __init__(self, filename):
        """Initialize instance from file."""
        self.filename = filename = os.fspath(filename)
        self._fh = open(filename, 'rb')
        if self._fh.read(8) != b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            self.close()
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

        self.byteorder = {0xFFFE: '<', 0xFEFF: '>'}[byteorder]
        if self.byteorder != '<':
            raise NotImplementedError('big-endian byte order not supported')

        if self.clsid == b'\x00' * 16:
            self.clsid = None
        if self.clsid is not None:
            raise OifFileError(f'cannot handle clsid {self.clsid}')

        if self.version_minor != 0x3E:
            raise OifFileError(
                f'cannot handle version_minor {self.version_minor}'
            )
        if mini_sector_shift != 0x0006:
            raise OifFileError(
                f'cannot handle mini_sector_shift {mini_sector_shift}'
            )
        if not (
            (self.version_major == 0x4 and sector_shift == 0x000C)
            or (
                self.version_major == 0x3
                and sector_shift == 0x0009
                and self.dir_len == 0
            )
        ):
            raise OifFileError(
                f'cannot handle version_major {self.version_major} '
                f'and sector_shift {sector_shift}'
            )

        self.sec_size = 2 ** sector_shift
        self.short_sec_size = 2 ** mini_sector_shift

        secfmt = '<' + ('I' * (self.sec_size // 4))
        # read DIFAT
        self._difat = list(
            struct.unpack('<' + ('I' * 109), self._fh.read(436))
        )
        nextsec = self.difat_start
        for i in range(self.difat_len):
            if nextsec >= CompoundFile.MAXREGSID:
                raise OifFileError('nextsec >= CompoundFile.MAXREGSID')
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
            raise OifFileError(f"no root directory found, got {root.name}")
        if root.create_time is not None:  # and root.modify_time is None
            raise OifFileError(f'invalid root.create_time {root.create_time}')
        if root.stream_size % self.short_sec_size != 0:
            raise OifFileError(
                f'root.stream_size {root.stream_size} does not match '
                f'short_sec_size {self.short_sec_size}'
            )
        # read mini stream
        self._ministream = b''.join(self._sec_chain(root.sector_start))
        self._ministream = self._ministream[: root.stream_size]
        # map stream/file names to directory entries
        nostream = CompoundFile.NOSTREAM
        join = '/'.join  # os.path.sep.join
        dirs = self._dirs
        visited = [False] * len(self._dirs)

        def parse(dirid, path):
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

    def _read_stream(self, direntry):
        """Return content of stream."""
        if direntry.stream_size < self.mini_stream_cutof_size:
            result = b''.join(self._mini_sec_chain(direntry.sector_start))
        else:
            result = b''.join(self._sec_chain(direntry.sector_start))
        return result[: direntry.stream_size]

    def _sec_read(self, secid):
        """Return content of sector from file."""
        self._fh.seek(self.sec_size + secid * self.sec_size)
        return self._fh.read(self.sec_size)

    def _sec_chain(self, secid):
        """Return iterator over FAT sector chain content."""
        while secid != CompoundFile.ENDOFCHAIN:
            if secid <= CompoundFile.MAXREGSECT:
                yield self._sec_read(secid)
            secid = self._fat[secid]

    def _mini_sec_read(self, secid):
        """Return content of sector from mini stream."""
        pos = secid * self.short_sec_size
        return self._ministream[pos : pos + self.short_sec_size]

    def _mini_sec_chain(self, secid):
        """Return iterator over mini FAT sector chain content."""
        while secid != CompoundFile.ENDOFCHAIN:
            if secid <= CompoundFile.MAXREGSECT:
                yield self._mini_sec_read(secid)
            secid = self._minifat[secid]

    def files(self):
        """Return sequence of file names."""
        return self._files.keys()

    def direntry(self, name):
        """Return DirectoryEntry of filename."""
        return self._files[name]

    def open_file(self, filename):
        """Return stream as file like object."""
        return BytesIO(self._read_stream(self._files[filename]))

    def format_tree(self):
        """Return formatted string with list of all files."""
        return '\n'.join(natural_sorted(self.files()))

    def close(self):
        """Close file handle."""
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self):
        """Return string with information about CompoundFile."""
        s = [
            self.__class__.__name__,
            os.path.normpath(os.path.normcase(self.filename)),
        ]
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
        ):
            s.append(f'{attr}: {getattr(self, attr)}')
        return '\n '.join(s)


class DirectoryEntry:
    """Compound Document Directory Entry."""

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

    def __init__(self, data, version_major):
        """Initialize directory entry from 128 bytes."""
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
        ) = struct.unpack('<64sHBBIII16sIQQIQ', data)

        if version_major == 3:
            self.stream_size = struct.unpack('<I', data[-8:-4])[0]
        if self.clsid == b'\000' * 16:
            self.clsid = None

        if name_len % 2 != 0 or name_len > 64:
            raise OifFileError(f'invalid name_len {name_len}')
        if self.color not in (0, 1):
            raise OifFileError(f'invalid color {self.color}')

        self.name = name[: name_len - 2].decode('utf-16')
        self.create_time = filetime(create_time)
        self.modify_time = filetime(modify_time)
        self.is_stream = self.entry_type == 2
        self.is_storage = self.entry_type == 1

    def __str__(self):
        """Return string with information about DirectoryEntry."""
        s = [self.__class__.__name__]
        for attr in self.__slots__[1:]:
            s.append(f'{attr}: {getattr(self, attr)}')
        return '\n '.join(s)


def format_dict(
    adict,
    prefix=' ',
    indent=' ',
    bullets=('', ''),
    excludes=('_',),
    linelen=79,
    trim=1,
):
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


def astype(value, types=None):
    """Return argument as one of types if possible."""
    if value[0] in '\'"':
        return value[1:-1]
    if types is None:
        types = int, float, str
    for typ in types:
        try:
            return typ(value)
        except (ValueError, TypeError, UnicodeEncodeError):
            pass
    return value


def filetime(ft):
    """Return Python datetime from Microsoft FILETIME number."""
    if not ft:
        return None
    sec, nsec = divmod(ft - 116444736000000000, 10000000)
    return datetime.utcfromtimestamp(sec).replace(microsecond=(nsec // 10))


if __name__ == '__main__':
    import doctest
    import tempfile  # noqa: used in doctrings

    numpy.set_printoptions(suppress=True, precision=5)
    doctest.testmod(optionflags=doctest.ELLIPSIS)
