# FUTURIST Framework

FUTURIST (FUture Temperatures Using River hISTory) is a modeling framework that predicts the likely impact of future dam operations on the river temperature 
(termed as thermal pollution).

The framework trains an Artificial Neural Network (ANN)model to learn historical patterns of the impact on rivers due to dam operations. 
Various dam characteristics, hydrology, topography and climate of the reservoir basin were used for training. 
A feedforward ANN model was selected, and hyperparameter tuning was performed for designing the network architecture.
The model consists of three dense layers with 256, 16 and 4 nodes, while the input layer contained seven nodes.
The training was performed over dams in US and then validated over selected sites in Southeast Asia. 


The output of the model is thermal regime change in river temperature, or difference between upstream and downstream river temperatures. 
There are qualitative classes assigned to this thermal change, labeled as severe/moderate wamring or cooling of tailwaters.

For questions or suggestions, please contact skahmad@uw.edu
