import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def load_data():
    x_trans = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\x_trans.mat", dtype=np.float64)
    y_trans = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\y_trans.mat", dtype=np.float64)
    z_trans = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\z_trans.mat", dtype=np.float64)
    room_corners = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\room_corners.mat", dtype=np.float64)

    return x_trans, y_trans, z_trans, room_corners

def plot_movement():
    x_trans, y_trans, z_trans, room_corners = load_data()
    t = np.arange(0, 7200, 1)

    first_row = room_corners[:, 0]

    fig, ax = plt.subplots(3, figsize=(10, 6))

    ax[0].plot(t, x_trans)
    ax[1].plot(t, y_trans)
    ax[2].plot(t, z_trans)
    
    fig_2 = plt.figure()
    ax_2 = plt.axes(projection='3d')
    ax_2.plot3D(x_trans, y_trans, z_trans)
    
    #ax_2.axes(projection='3d')
    #ax_2.plot3D(x_trans, y_trans, z_trans)
    #fig_2, ax_2 = plt.subplots(figsize=(10, 6))

    plt.savefig(r"Task sheet files-20221008\Affine\Affine\x_data.jpeg", dpi=300)
    plt.show()

def plot_head():
    chamod_head = plt.imread(r"Task 2 Images\Chamod head.jpeg")

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.imshow(chamod_head)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.show()

    position = ax.get_position()
    ax.set_position()

#plot_head()
plot_movement()

print('Hello world')