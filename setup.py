from setuptools import setup, find_packages

setup(
    name="sbayes",
    version="0.1",
    description="MCMC algorithms to identify linguistic contact zones",
    author="Peter Ranacher",
    author_email="peter.ranacher@geo.uzh.ch",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='data linguistics',
    license='???',
    url="MCMC algorithms to identify linguistic contact zones",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
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
