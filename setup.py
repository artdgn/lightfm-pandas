from setuptools import setup, find_packages
from os import path

version = '0.2.0'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lightfm_pandas',
    version=version,
    description='LightFM convenience tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/artdgn/lightfm-pandas',
    author='Arthur Deygin',
    author_email='arthurdgn@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='recommendations machine learning',
    packages=find_packages(exclude=['tests', 'examples']),
    install_requires=open('requirements.in').read().split('\n'),
)
