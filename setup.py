from setuptools import setup, find_packages

setup(
   name='supmatch'
   version='2024.7.0',
   description='Supervised Matching',
   author='Jacob Goldsmith',
   author_email='jacobg314@hotmail.com',
   packages=find_packages(),  #same as name
   install_requires=['numpy', 'pandas', 'scikit-learn'], #external packages as dependencies
)
