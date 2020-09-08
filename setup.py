from setuptools import setup, find_packages

setup(
    name="sbayes",
    version="0.1",
    description="MCMC algorithms to identify contact areas in cultural data",
    author="Peter Ranacher, Nico Neureiter",
    author_email="peter.ranacher@geo.uzh.ch",
    #long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='data linguistics',
    license='???',
    url="https://github.com/derpetermann/sbayes",
    packages=find_packages(where='sbayes'),
    package_dir={'':''},
    platforms='any',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",],
    entry_points={
        'console_scripts': [
            'sbayes = sbayes.cli:main',
        ]
    }
)
