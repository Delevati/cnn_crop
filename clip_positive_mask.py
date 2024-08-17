import fiona
import rasterio
import rasterio.mask
import os
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from shapely.geometry import mapping
import geopandas as gpd
import numpy as np

src_raster_path = os.getenv('IMAGE_PATH')
shp_file_path = os.getenv('CROPS_SHAPE_PATH')
shape_path = os.getenv('SHAPE_PATH')
output_dir = 'positive_crop'
block_size = 1024

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_block(i, j, out_image, out_transform, out_meta):
    block_height = min(block_size, out_image.shape[1] - i)
    block_width = min(block_size, out_image.shape[2] - j)

    window = Window(j, i, block_width, block_height)

    block = out_image[:, i:i + block_height, j:j + block_width]

    if np.any(block):
        block_transform = out_transform * rasterio.Affine.translation(-j * out_transform[0], -i * out_transform[4])

        block_meta = out_meta.copy()
        block_meta.update({
            "height": block_height,
            "width": block_width,
            "transform": block_transform
        })

        block_output_path = os.path.join(output_dir, f"block_{i}_{j}.tif")

        with rasterio.open(block_output_path, "w", **block_meta) as dest:
            dest.write(block)

def main():
    crops_gdf = gpd.read_file(shp_file_path)
    
    if shape_path:
        shape_gdf = gpd.read_file(shape_path)
        combined_gdf = gpd.overlay(crops_gdf, shape_gdf, how='intersection')
    else:
        combined_gdf = crops_gdf

    shapes = [mapping(geom) for geom in combined_gdf.geometry]

    with rasterio.open(src_raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
        
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        height, width = out_image.shape[1], out_image.shape[2]

        tasks = [(i, j) for i in range(0, height, block_size) for j in range(0, width, block_size)]

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(tqdm(executor.map(lambda args: process_block(*args, out_image, out_transform, out_meta), tasks),
                      total=len(tasks),
                      desc="Processando blocos"))

if __name__ == "__main__":
    main()
