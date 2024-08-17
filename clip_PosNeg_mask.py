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
block_size = 256

positive_output_dir = 'positive_blocks'
negative_output_dir = 'negative_blocks'
os.makedirs(positive_output_dir, exist_ok=True)
os.makedirs(negative_output_dir, exist_ok=True)

def process_block(i, j, out_image, out_transform, out_meta, output_dir):
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

def process_positive_blocks():
    crops_gdf = gpd.read_file(shp_file_path)

    if shape_path:
        shape_gdf = gpd.read_file(shape_path)
        combined_gdf = gpd.overlay(crops_gdf, shape_gdf, how='intersection')
    else:
        combined_gdf = crops_gdf

    shapes = [mapping(geom) for geom in combined_gdf.geometry]

    with rasterio.open(src_raster_path) as src:
        out_image_pos, out_transform_pos = rasterio.mask.mask(src, shapes, crop=True)
        out_meta_pos = src.meta

        out_meta_pos.update({
            "driver": "GTiff",
            "height": out_image_pos.shape[1],
            "width": out_image_pos.shape[2],
            "transform": out_transform_pos
        })

        height, width = out_image_pos.shape[1], out_image_pos.shape[2]

        tasks_pos = [(i, j) for i in range(0, height, block_size) for j in range(0, width, block_size)]

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(tqdm(executor.map(lambda args: process_block(*args, out_image_pos, out_transform_pos, out_meta_pos, positive_output_dir), tasks_pos),
                      total=len(tasks_pos),
                      desc="Processando blocos positivos"))

def process_negative_blocks():
    crops_gdf = gpd.read_file(shp_file_path)
    
    if shape_path:
        shape_gdf = gpd.read_file(shape_path)
        combined_gdf = gpd.overlay(crops_gdf, shape_gdf, how='intersection')
    else:
        combined_gdf = crops_gdf

    shapes = [mapping(geom) for geom in combined_gdf.geometry]

    with rasterio.open(src_raster_path) as src:
        out_image_neg, out_transform_neg = rasterio.mask.mask(src, shapes, invert=True)
        out_meta_neg = src.meta

        out_meta_neg.update({
            "driver": "GTiff",
            "height": out_image_neg.shape[1],
            "width": out_image_neg.shape[2],
            "transform": out_transform_neg
        })

        height, width = out_image_neg.shape[1], out_image_neg.shape[2]

        tasks_neg = [(i, j) for i in range(0, height, block_size) for j in range(0, width, block_size)]

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(tqdm(executor.map(lambda args: process_block(*args, out_image_neg, out_transform_neg, out_meta_neg, negative_output_dir), tasks_neg),
                      total=len(tasks_neg),
                      desc="Processando blocos negativos"))

def main():
    process_positive_blocks()
    process_negative_blocks()

if __name__ == "__main__":
    main()
