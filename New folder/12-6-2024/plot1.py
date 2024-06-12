import numpy as np
import matplotlib.pyplot as plt



#dataset
x = [1,2,3,4,5]
y = [2,3,5,7,11]

#create a plot for our data
plt.plot (x,y)

#customization for the plot

#add a tittle
plt.title('lineplot')

#add the label
plt.xlabel("x-axis")
plt.ylabel("y-axis")


#output the plot
plt.show()