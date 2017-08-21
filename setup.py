from setuptools import setup
import setuptools

print(setuptools.find_packages())
setup(name='koko-driver',
      version='0.1',
      description='The best project in the world',
      url='https://github.com/Mrgemy95/GP',
      author='KOKO-Mind',
      license='MIT',
      packages=setuptools.find_packages(),
      zip_safe=False)