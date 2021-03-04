from setuptools import setup, find_packages

setup(
    name="sbayes",
    version="1.0",
    description="MCMC algorithms to identify contact areas in cultural data",
    author="Peter Ranacher, Nico Neureiter",
    author_email="peter.ranacher@geo.uzh.ch",
    #long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='data linguistics',
    license='???',
    url="https://github.com/derpetermann/sbayes",
    package_dir={'sbayes': 'sbayes'},
    packages=find_packages(),
    platforms='any',
    include_package_data=True,
    package_data={
        'sbayes.config': ['default_config.json',
                          'default_config_plot.json',
                          'default_config_simulation.json'],
    },
    install_requires=[
        "descartes",
        "geopandas",
        "matplotlib",
        "numpy",
        "pandas",
        "pyproj",
        "scipy",
        "Shapely",
        "seaborn",
        "fastcluster",
        "typing_extensions",
        "pycldf",
    ],
    entry_points={
        'console_scripts': [
            'sbayes = sbayes.cli:main',
        ]
    }
)
