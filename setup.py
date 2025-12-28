from setuptools import setup, find_packages

setup(
    name="jaxrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "flax>=0.8.0",
        "optax>=0.1.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)