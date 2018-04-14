from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

learning_rate = [0.01, 0.05, 0.25, 0.1, 0.5, 0.01, 0.05, 0.25, 0.1, 0.5, 0.01, 0.05, 0.25, 0.1, 0.5]
iteration = [10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1]
accuracy = [0.985, 0.988, 0.9926, 0.9914, 0.9911, 0.984, 0.988, 0.990, 0.9887, 0.99, 0.965, 0.980, 0.988, 0.987, 0.989]

xdata = learning_rate
ydata = iteration
zdata = accuracy

ax.set_xlabel('learning rate')
ax.set_ylabel('iteration')
ax.set_zlabel('accuracy')

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='autumn')

plt.show()