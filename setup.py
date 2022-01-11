import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="candle",
    version="0.0.1",
    author="Sri Pranav Kunda",
    author_email="sripkunda@gmail.com",
    description="A tiny, pedagogical neural network library with a pytorch-like API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sripkunda/candle/",
    project_urls={
        "Bug Tracker": "https://github.com/sripkunda/candle/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "candle"},
    packages=setuptools.find_packages(where="candle"),
    python_requires=">=3.6",
)
