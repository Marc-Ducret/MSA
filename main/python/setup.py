from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='minecraft_rl',
      packages=[package for package in find_packages()
                if package.startswith('minecraft')],
      install_requires=[
      ],
      description='Minecraft environment for RL',
      author='Marc Ducret',
      url='https://github.com/VengeurK/MSA',
      author_email='marc.ducret@ens.fr',
      version='1.0')
