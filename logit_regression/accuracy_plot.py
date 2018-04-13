from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

learning_rate = [0.001, 0.01, 0.1, 0.1, 0.01, 0.01]
iteration = [1, 1, 1, 5, 5, 10]
accuracy = [1, 1, 0.538, 0.48, 0.6645, 0.54]

xdata = learning_rate
ydata = iteration
zdata = accuracy

ax.set_xlabel('learning rate')
ax.set_ylabel('iteration')
ax.set_zlabel('accuracy')

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='copper')

plt.show()