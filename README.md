## Introduction

Existing 3D shape datasets in the research community are generally limited to objects or scenes at the room or street level. City level shape datasets are rare due to the difficulty in data collection and processing. However, city level datasets are important as they present a new category of 3D data - one with a high variance in geometric complexity and size. Residential buildings, skyscrapers, historical buildings, and modern commercial buildings all present unique geometric structures and styles. This work focuses on collecting city level object data, demonstrating the unique geometric properties of the dataset, and proposing new research directions and benchmarks for city generation. To that end, we collect over 1,000,000 geo-referenced objects for New York City and Zurich. We benchmark the performance of various state-of-the-art methods for two challenging tasks: (1) city layout generation, and (2) building shape generation. Moreover, we propose an auto-encoding tree neural network for 2D building footprint and 3D building cuboid generation. The dataset, tools, and algorithms will be released to the community.

This repository is contains the tools and data for Real Urban Structure data in multiple formats.

## Authors
Congcong Wen, Wenyu Han, Lazarus Chok, Yan Liang Tan, Sheung Lung Chan, Hang Zhao, Chen Feng,

## Paper 
[link](https://scene-understanding.com/papers/RealCity3D.pdf)

## Datasets
The current release contains the point cloud and mesh forms of the cities below. The data is geo referenced to facilitate "City Level" tasks.

| City | Country | Continent | Raw Meshes | Voxel | Triangulated Meshes | Point Cloud | DEM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| New York City | United States | North America | [download](https://drive.google.com/drive/folders/1e11MGq9BYS8fFdbDUWBtaL3cLRZ2aFNT?usp=sharing) | [download](https://drive.google.com/file/d/1wAGWK9M-jMlNbXsurosXjRqbj6Hmfk7N/view?usp=sharing)|[download](https://drive.google.com/drive/folders/1v0ZiqweL3mX82Qa7OEPKY8DRwCS5mj87?usp=sharing) |[download](https://drive.google.com/open?id=1b1edO0_zgSlwnnDfH7S9dcuNvsFHtM1A)| |
| Zurich | Switzerland | Europe | [download](https://drive.google.com/drive/folders/17OJtbjm3sJxIIaZSdc_UEVnUC8PuxSuy?usp=sharing) | | [download](https://drive.google.com/drive/folders/17IRpVipV9Y7l3v8YGATem6juTmQkDABq?usp=sharing) |[download](https://drive.google.com/drive/folders/1zsojfGOkraFhMe76t1osJ8gHJs9KT7EA?usp=sharing) | |


[NYC 64^3 voxel](https://drive.google.com/open?id=1KAPIj_Zyu6htnEebHnmAOGjfZliML2U-)
## Tools

Following are the tools we have released or are planning to release:

- **Geo-Location Extractor**: Python tool to extract geolocations of structures from the mesh files. (GeoProcessing.py)

- **Search Tool**: Python tool for retreiving all structures in a bounding box. (GeoProcessing.py)

- **Visualization Tool**: Tool for vizualising all structures. (Planned Release)

### Geo Processing and Search Tool

The Geo Processing tool uses the raw mesh data to extract the geo locations. The extracted locations are then used for searching through the files.

#### How to run?

The tool has two modes, searching and extraction. The default is search. It needs the location of the index.csv file containing the geo locations, the Northing and Easting values of the top left and bottom right points of the search bounded box

The Extraction mode needs the location of the raw mesh files.

These tools have been tested for python 3.6.8

### Visualization Tool
Planned for release

## Contact

Please contact Congcong Wen at cw3437 at nyu dot edu or Wenyu Han at wh1264 at nyu dot edu for questions regarding this project.
