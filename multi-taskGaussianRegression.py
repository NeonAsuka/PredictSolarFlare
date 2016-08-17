import numpy as np

# Import the Necessary Modules
# GPRegression module
from numpy import gpregressor
from numpy import abalone_data
 
# Plotting modules
import matplotlib
from pylab import plot, show, ylim, yticks
import matplotlib.pyplot as plt

def graph_abalone_data(y,x,title,fignum=None):
    att_names = ["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]
    if fignum is None:
        plt.figure()
    else:
        plt.figure(fignum)
    for i in range(0,len(att_names)):
        plt.subplot(4,2,i)
        plt.scatter(x[:,i+3],y, faceted=False)
        plt.xlabel(att_names[i])
    #plt.title(title)
    plt.show()

# Load and process the dataset
raw,cooked = abalone_data.get_data()

# plot the data, to get some idea about how it looks
y = cooked[0]
x = cooked[1]
#graph_abalone_data(y,x,"Cooked",1)
#graph_abalone_data(raw[0],raw[1],"Raw",2)

num_partions = 5
# Initialize the regressor
gpr = GPRegressor(x,y,num_partions)
