import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="candl",
    version="0.0.3",
    author="Sri Pranav Kunda",
    author_email="sripkunda@gmail.com",
    description="A tiny, pedagogical neural network library with a pytorch-like API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sripkunda/candl/",
    project_urls={
        "Bug Tracker": "https://github.com/sripkunda/candl/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "candl"},
    packages=setuptools.find_packages(where="candl"),
    python_requires=">=3.6",
)
