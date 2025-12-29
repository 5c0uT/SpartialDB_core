from setuptools import setup, find_packages

setup(
    name='spatial_db',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'spatial_db': ['lib/*.pyd', 'lib/*.dll'],  # Включаем нативные библиотеки
    },
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'fsspec>=2022.1.0',
        'tqdm>=4.62.0',
        'matplotlib>=3.5.0',
    ],
    entry_points={
        'console_scripts': [
            'spatialdb-benchmark = spatial_db.examples.benchmark:run_benchmark',
        ],
    },
    python_requires='>=3.8',
    author='Your Name',
    author_email='your.email@example.com',
    description='GPU-accelerated spatial database with PhysX',
    keywords='gpu spatial database physx 3d',
)