## Introduction

Existing 3D shape datasets foster 3D deep learning research in the vision, graphics, and robotics communities by motivating research, specifying challenges, and enabling model comparisons. However, most existing 3D shape datasets are comprised of CAD models or point clouds cans at either object-level or room-level, leaving out a large source of 3D shape data: real-world cities. Cities are important because they contain complex shapes such as skyscrapers, residential buildings, roads, and bridges. These shapes contain rich details that can be significantly different from object-level and room-level 3D shapes. Such inherent domain differences bring challenges to existing deep learning methods on 3D data, especially unsupervised ones, therefore inviting additional research in this area. In this work, we collect and process more than 1,000,000 georeferenced 3D shapes from the city of New York and the city of Zurich,and demonstrate the performance gap of three unsupervised 3D deep learning methods on our dataset and existing datasets. We are also actively working to include other major world cities and benchmarking more 3D deep learning methodson this dataset. 

This repository is contains the tools and data for Real Urban Structure data in multiple formats.

## Authors
Tanay Varshney, Yuqiong Li, Ruoyu Wang, Xuchu Xu, Hang Zhao, Zhiding Yu, Chen Feng

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

Please contact Tanay Varshney at tanay at nyu dot edu or Yuqiong Li at yl5090 at nyu dot edu for questions regarding this project.
