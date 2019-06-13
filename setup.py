from setuptools import setup, find_packages
from nlpsota import VERSION, AUTHOR

setup(name='nlpsota',
      version=VERSION,
      description='An NLP Libraries',
      url='https://github.com/yinchuandong/NLP-SOTA',
      author=AUTHOR,
      license='MIT',
      packages=find_packages(include='nlpsota'),
      install_requires=[
          'torch',
          'torchtext',
          'pytorch-pretrained-bert',
          'numpy',
          'sklearn',
          'pandas',
      ],
      zip_safe=True)
