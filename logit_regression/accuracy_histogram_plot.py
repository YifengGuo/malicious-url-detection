from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

learning_rate = [0.01, 0.05, 0.25, 0.1, 0.5, 0.01, 0.05, 0.25, 0.1, 0.5, 0.01, 0.05, 0.25, 0.1, 0.5]
iteration = [10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1]
accuracy = [0.985, 0.988, 0.9926, 0.9914, 0.9911, 0.984, 0.988, 0.990, 0.9887, 0.99, 0.965, 0.980, 0.988, 0.987, 0.989]

xpos = learning_rate
ypos = iteration
zpos = np.zeros(15)

dx = 0.02 * np.ones(15)
dy = np.ones(15)
dz = accuracy
ax1.set_xlabel('learning rate')
ax1.set_ylabel('iteration')
ax1.set_zlabel('accuracy')
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'green')
plt.show()