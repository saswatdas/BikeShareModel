#This is what your ‘setup.py’ file should look like.
 
#from setuptools import setup, find_packages

#setup(
 #   name="bikeshare_model", #Name
  #  version="0.0.1", #Version
   # include_package_data=True,
    #packages=find_packages(exclude=['docs', 'tests*']),  # Automatically find the packages that are recognized in the '__init__.py'.
    
#)

from setuptools import setup, find_packages 

# Package metadata 
name = 'bikeshare_model'
version = '0.0.1'
description = 'bikeshare model that predicts the bike count'

# Package setup 
setup( 
	name=name, 
	version=version, 
	description=description, 
	packages=find_packages(), 
    package_data={
      '': ['*.*'],
      '': ['VERSION']
      #'': ['*.csv']     
   },
   include_package_data=True,
) 
