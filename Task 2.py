import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from loading_data import load_affine_data

def plot_movement(show_trans_coors):
    x_trans, y_trans, z_trans, room_corners, head, alpha, phi, theta = load_affine_data()
    
    if show_trans_coors == True:
        t = np.arange(0, 7200, 1)

        fig, ax = plt.subplots(3, figsize=(10, 6))

        ax[0].plot(t, x_trans)
        ax[1].plot(t, y_trans)
        ax[2].plot(t, z_trans)
        plt.savefig(r"Task 2 Images\x_data.jpeg", dpi=300)
        plt.show()

    #ax_2.axes(projection='3d')
    #ax_2.plot3D(x_trans, y_trans, z_trans)
    #fig_2, ax_2 = plt.subplots(figsize=(10, 6))

def plot_head(show_trans, show_head, show_room):
    x_trans, y_trans, z_trans, room_corners, head, alpha, phi, theta = load_affine_data()
    
    room_x, room_y, room_z = room_corners[:, 0], room_corners[:, 1], room_corners[:, 2]
    head_x, head_y, head_z = head[:, 0], head[:, 1], head[:, 2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if show_trans == True:
        ax.plot3D(x_trans, y_trans, z_trans, 'k*')
    
    if show_head == True:
        ax.plot3D(head_x, head_y, head_z, 'b.')

    if show_room == True:
        ax.plot3D(room_x, room_y, room_z, 'k.')
        
    plt.show()

def show_animation(show_room):
    x_trans, y_trans, z_trans, room_corners, head, alpha, phi, theta = load_affine_data()
    
    room_x, room_y, room_z = room_corners[:, 0], room_corners[:, 1], room_corners[:, 2]
    head_x, head_y, head_z = head[:, 0], head[:, 1], head[:, 2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.plot3D(head_x, head_y, head_z, 'b.')
    
    if show_room == True:
        ax.plot3D(room_x, room_y, room_z, 'k.')
    

    plt.pause(1)
    N = np.size(x_trans)
    
    for i in range(N):
        head_x = head_x + x_trans[i]
        head_y = head_y + y_trans[i]
        head_z = head_z + z_trans[i]
    
        ax.plot3D(head_x, head_y, head_z, 'b.')
        plt.pause(0.001)
        

#plot_movement(show_trans_coors=True)
#plot_head(show_trans=False, show_head=True, show_room=True)
show_animation(show_room=True)

print('Hello world')