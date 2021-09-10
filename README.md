#This is a repository for intelligence_building of refrigeration.
Here are some python package which can be used for developing a online fault detection and diagnosis model for chiller.
please use git clone to copy the repository.
usage:
	from intelligence_building import *

requirement:
	numpy
	pandas
	torch

#Basic logic of intelligence_building package
1. get a dataset using **dataset** module
2. considering the usage to choose the **traditional** machine learning model or **deep learning** model
3. using metrics to analyze the result
4. using **log** module to write infomation to the log file
