from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alphazero-strands",
    version="0.1.0",
    author="John J Boren",
    author_email="jboren@alaska.edu",
    description="Integration of AlphaZero with AWS STRANDS Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnJBoren/alphazero-strands-integration",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.8.0",
        "strands>=0.1.0",
        "strands-agents-tools>=0.1.0",
        "matplotlib>=3.4.0",
        "requests>=2.25.0",
        "huggingface-hub>=0.16.0",
        "python-chess>=1.9.0",
    ],
    entry_points={
        "console_scripts": [
            "alphazero-train=examples.chess_training:main",
            "alphazero-visualize=examples.connect4_visualization:main",
            "alphazero-analyze=examples.multi_agent_analysis:main",
        ],
    },
)