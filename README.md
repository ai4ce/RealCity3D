## Introduction

Existing 3D shape datasets in the research community are generally limited to objects or scenes at the room or street level. City level shape datasets are rare due to the difficulty in data collection and processing. However, city level datasets are important as they present a new category of 3D data - one with a high variance in geometric complexity and size. Residential buildings, skyscrapers, historical buildings, and modern commercial buildings all present unique geometric structures and styles. This work focuses on collecting city level object data, demonstrating the unique geometric properties of the dataset, and proposing new research directions and benchmarks for city generation. To that end, we collect over 1,000,000 geo-referenced objects for New York City and Zurich. We benchmark the performance of various state-of-the-art methods for two challenging tasks: (1) city layout generation, and (2) building shape generation. Moreover, we propose an auto-encoding tree neural network for 2D building footprint and 3D building cuboid generation. The dataset, tools, and algorithms will be released to the community.

This repository is contains the tools and data for Real Urban Structure data in multiple formats.

## Authors
Wenyu Han, Congcong Wen*, Lazarus Chok, Yan Liang Tan, Sheung Lung Chan, Hang Zhao, Chen Feng
*Corresponding author. Email: wencc@nyu.edu

## Datasets
The current release contains the point cloud and mesh forms of the cities below. The data is geo referenced to facilitate "City Level" tasks.

| City | Country | Continent | Polygon Meshes | Voxel | Triangulated Meshes | Point Cloud |
| --- | --- | --- | --- | --- | --- | --- |
| New York City | United States | North America | [download](https://drive.google.com/drive/folders/1XAfaWw0NRgJRefyYgItc_AI7ahEo_HVj?usp=sharing) | [download](https://drive.google.com/drive/folders/10_8PQ4SAwXv8VHM3yzo1A6k7DSI89gC0?usp=sharing)|[download](https://drive.google.com/drive/folders/1kVeAs3lsLND5xVWjYmsyNxID7xIZsAkt?usp=sharing) |[download](https://drive.google.com/drive/folders/10FN8wcDdY5u8XC6Gxm1BbK6EvZCjlR4y?usp=sharing)| |
| Zurich | Switzerland | Europe | [download](https://drive.google.com/drive/folders/1t5Vo8eNbzqW2KL1KeDs07j6khSgG4t2r?usp=sharing) | [download](https://drive.google.com/drive/folders/1Y93S-QuqfamhvIOmJvRu0HhuEYyv3yW_?usp=sharing) | [download](https://drive.google.com/drive/folders/1lDtRJSznBw2jc_px1yrZ-f_zDboGeonU?usp=sharing) |[download](https://drive.google.com/drive/folders/1hsrvLJMTwz2KaDdeSTTUCiTy-VPzJqUX?usp=sharing) | |

## [Website](https://ai4ce.github.io/RealCity3D/) 

## Tools

Following are the tools we have released:

- **Geo-Location Extractor**: Python tool to extract geolocations of structures from the mesh files. (GeoProcessing.py)

- **Search Tool**: Python tool for retreiving all structures in a bounding box. (GeoProcessing.py)



### Geo Processing and Search Tool

The Geo Processing tool uses the raw mesh data to extract the geo locations. The extracted locations are then used for searching through the files.

#### How to run?

The tool has two modes, searching and extraction. The default is search. It needs the location of the index.csv file containing the geo locations, the Northing and Easting values of the top left and bottom right points of the search bounded box

The Extraction mode needs the location of the raw mesh files.

These tools have been tested for python 3.6.8



## Contact

Please contact Congcong Wen at cw3437 at nyu dot edu or Wenyu Han at wh1264 at nyu dot edu for questions regarding this project.
