from setuptools import find_packages, setup

setup(
    name="grid_uhi_mask",
    version="2.0.1",
    packages=find_packages(),
    install_requires=["numpy>=1.24", "xarray>=2023.1", "netCDF4>=1.6"],
    author="Gandome Mayeul Quenum",
    description="MOD_Mask v2: grid-based rural-reference selection and UHI calculation",
    url="https://github.com/Gandome/rural_mask_grid_city_crop",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)
