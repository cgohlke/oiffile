Read Olympus(r) image files (OIF and OIB)
=========================================

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
