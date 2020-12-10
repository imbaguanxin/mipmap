# mipmap

Sample to run:

```shell
activatedaisy
cd /n/groups/htem/Segmentation/xg76/mipmap
python zarr_tiff.py config.json
```

content in config file:

- zarr_file: the path to the zarr file we want to extract
- section: the section we want to use (just a name, not affecting the program)
- raw_ds: path to raw_ds
- coord_begin: the upper left coordinate, with least z
- coord_end: the botom right coordinate, with biggest z
- interval: the interval we want to skip.
    For example, the z selection would be range(coord_begin.z, coord_end.z, interval)
    or: [coord_begin.z, coord_begin.z + interval, coord_begin.z + 2interval + .... ]
- output_folder: the location to store mipmap. This folder is created if not exist.