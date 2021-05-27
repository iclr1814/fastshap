import setuptools

setuptools.setup(
    name="fastshap",
    version="0.0.1",
    author="",
    author_email="",
    description="For amortizing local Shapley value explanations.",
    long_description="""
        Hello world!
    """,
    long_description_content_type="text/markdown",
    url="",
    packages=['fastshap_torch'],
    install_requires=[
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
