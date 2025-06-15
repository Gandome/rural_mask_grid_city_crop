from setuptools import setup, find_packages

setup(
    name="climate_city_stats",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "xarray", "geopandas", "numpy", "pandas", "matplotlib", "shapely", "tqdm", "netCDF4"
    ],
    author="Your Name",
    description="City-based climate statistics from gridded datasets",
    url="https://github.com/Gandome/climate_city_stats",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.8',
)

