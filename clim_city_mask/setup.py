from setuptools import find_packages, setup

setup(
    name="climate_city_stats",
    version="2.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "xarray>=2023.1",
        "netCDF4>=1.6",
        "geopandas>=0.13",
        "matplotlib>=3.7",
        "shapely>=2.0",
        "tqdm>=4.65",
    ],
    author="Gandome Mayeul Quenum",
    description="City extraction and visualization tools for MOD_Mask/UHI v2",
    url="https://github.com/Gandome/rural_mask_grid_city_crop",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)
