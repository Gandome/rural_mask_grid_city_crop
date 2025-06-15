from setuptools import setup, find_packages

setup(
    name="grid_uhi_mask",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "xarray", "numpy", "pandas", "scikit-learn"
    ],
    author="Gandome Mayeul QUenum",
    description="Grid-based UHI calculation from gridded datasets",
    url="https://github.com/Gandome/grid_uhi_mask",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.8',
)

