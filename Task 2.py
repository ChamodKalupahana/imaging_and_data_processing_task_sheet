import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from loading_data import load_affine_data

def plot_head(show_trans, show_head, show_room):
    """plot_head shows the intial frame of the head

    Args:
        show_trans (Boolan): Plot the x, y, z data on a separate figure
        show_head (Boolan): Plot a 3D figure of the head coords
        show_room (Boolan): Shows the room corner points on the 3D figure
    """
    
    # extracts downloaded data into separate variables
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

def rotation_matrix(theta, phi, alpha ,new_head_x, new_head_y, new_head_z, centre_of_mass, i):

    head_x, head_y, head_z = new_head_x, new_head_y, new_head_z
    # bring head coords back to origin
    centred_new_head_x = new_head_x - centre_of_mass[0]
    centred_new_head_y = new_head_y - centre_of_mass[1]
    centred_new_head_z = new_head_z - centre_of_mass[2]

    # 3D rotation matrices
    R_x = np.array([[1, 0, 0], [0, np.cos(theta[i]), -np.sin(theta[i])], [0, np.sin(theta[i]), np.cos(theta[i])]])
    R_y = np.array([[np.cos(phi[i]), 0, np.sin(phi[i])], [0, 1, 0], [-np.sin(phi[i]), 0, np.cos(phi[i])]])
    R_z = np.array([[np.cos(alpha[i]), -np.sin(alpha[i]), 0], [np.sin(alpha[i]), 1, 0], [-np.sin(alpha[i]), 0, np.cos(alpha[i])]])
    
    n = np.size(centred_new_head_x)
    new_coords = np.zeros([50, 3])

    for k in range(n):
        temp_coords = np.array([centred_new_head_x[k], centred_new_head_y[k], centred_new_head_z[k]])

        # apply rotations to each axis
        temp_coords = np.dot(R_x, temp_coords)
        temp_coords = np.dot(R_y, temp_coords)
        temp_coords = np.dot(R_z, temp_coords)

        # append all rotated coords to new array
        new_coords[k] = temp_coords

    new_head_x = new_coords[:,0] + centre_of_mass[0]
    new_head_y = new_coords[:,1] + centre_of_mass[1]
    new_head_z = new_coords[:,2] + centre_of_mass[2]
    
    return new_head_x, new_head_y, new_head_z

def rotation_old_matrix(theta, phi, alpha ,new_head_x, new_head_y, new_head_z, centre_of_mass, i):
    
    centred_new_head_x = new_head_x - centre_of_mass[0]
    centred_new_head_y = new_head_y - centre_of_mass[1]
    centred_new_head_z = new_head_z - centre_of_mass[2]

    centred_new_head = np.array([centred_new_head_x, centred_new_head_y, centred_new_head_z])

    R_x = np.array([[1, 0, 0], [0, np.cos(theta[i]), -np.sin(theta[i])], [0, np.sin(theta[i]), np.cos(theta[i])]])
    R_y = np.array([[np.cos(phi[i]), 0, np.sin(phi[i])], [0, 1, 0], [-np.sin(phi[i]), 0, np.cos(phi[i])]])
    R_z = np.array([[np.cos(alpha[i]), -np.sin(alpha[i]), 0], [np.sin(alpha[i]), 1, 0], [-np.sin(alpha[i]), 0, np.cos(alpha[i])]])

    rotated_new_head_x_total, rotated_new_head_y_total, rotated_new_head_z_total = np.array([]), np.array([]), np.array([])
    
    for k in range(50):
        centred_new_head_small = centred_new_head[:, k]

        rotated_new_head_x = np.dot(R_x, centred_new_head_small[0])
        rotated_new_head_y = np.dot(R_y, centred_new_head_small[1])
        rotated_new_head_z = np.dot(R_z, centred_new_head_small[2])

        rotated_new_head_x_total = np.append(rotated_new_head_x_total, rotated_new_head_x + centre_of_mass[0])
        rotated_new_head_y_total = np.append(rotated_new_head_y_total, rotated_new_head_y + centre_of_mass[1])
        rotated_new_head_z_total = np.append(rotated_new_head_z_total, rotated_new_head_z + centre_of_mass[2])


    return rotated_new_head_x_total, rotated_new_head_y_total, rotated_new_head_z_total

def translation_matrix(head_x, head_y, head_z, x_trans, y_trans, z_trans, i):
    
    # affine data shows the translational data from the origin
    new_head_x = head_x + x_trans[i]
    new_head_y = head_y + y_trans[i]
    new_head_z = head_z + z_trans[i]

    return new_head_x, new_head_y, new_head_z


def show_animation(affine_data,show_room, small_room, start_frame):
    # unpack affine data from loaded array
    x_trans, y_trans, z_trans = affine_data[0], affine_data[1], affine_data[2]
    room_corners, head = affine_data[3], affine_data[4]
    alpha, phi, theta = affine_data[5], affine_data[6], affine_data[7]

    # unpack room and head co-ords
    room_x, room_y, room_z = room_corners[:, 0], room_corners[:, 1], room_corners[:, 2]
    head_x, head_y, head_z = head[:, 0], head[:, 1], head[:, 2]

    # make blank figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.plot3D(head_x, head_y, head_z, 'b.')

    plt.pause(1)
    
    N = np.size(x_trans)
    centre_of_mass = np.array([np.average(head_x), np.average(head_y), np.average(head_z)])

    if show_room == True:
        if small_room == False:
            ax.plot3D(room_x, room_y, room_z, 'k.')
    
    if small_room == True:
        ax.plot3D(room_x / 3, room_y / 3, room_z / 3, 'k.')

    #animation_data, = ax.scatter3D(head_x, head_y, head_z, 'b.')
    ax.scatter3D(head_x, head_y, head_z, 'b.')
    
    for i in range(start_frame ,N):
        #new_head_x, new_head_y, new_head_z = rotation_matrix(theta, phi, alpha ,new_head_x, new_head_y, new_head_z, centre_of_mass, i)
        new_head_x, new_head_y, new_head_z = translation_matrix(head_x, head_y, head_z,x_trans, y_trans, z_trans, i)
        # points move 1 by 1? not all together

        ax.scatter3D(new_head_x, new_head_y, new_head_z, 'b.')
        #animation_data.set_xdata(new_head_x)
        #animation_data.set_ydata(new_head_y)
        #animation_data.set_zdata(new_head_z)

        #new_head_x, new_head_y, new_head_z = head_x, head_y, head_z
        plt.pause(0.0001)
        
        print('Loop', i)

def show_old_animation(affine_data,show_room, small_room, start_frame):
    # unpack affine data from loaded array
    x_trans, y_trans, z_trans = affine_data[0], affine_data[1], affine_data[2]
    room_corners, head = affine_data[3], affine_data[4]
    alpha, phi, theta = affine_data[5], affine_data[6], affine_data[7]

    # unpack room and head co-ords
    room_x, room_y, room_z = room_corners[:, 0], room_corners[:, 1], room_corners[:, 2]
    head_x, head_y, head_z = head[:, 0], head[:, 1], head[:, 2]

    if small_room == True:
        room_x  = room_x / 3
        room_y  = room_y / 3
        room_z  = room_z / 3

    # make blank figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    #ax.plot3D(head_x, head_y, head_z, 'b.')

    #plt.pause(1)
    
    N = np.size(x_trans)
    centre_of_mass = np.array([np.average(head_x), np.average(head_y), np.average(head_z)])

    
    for i in range(start_frame ,N):
        new_head_x, new_head_y, new_head_z = head_x, head_y, head_z

        new_head_x, new_head_y, new_head_z = rotation_matrix(theta, phi, alpha ,new_head_x, new_head_y, new_head_z, centre_of_mass, i)
        new_head_x, new_head_y, new_head_z = translation_matrix(new_head_x, new_head_y, new_head_z, x_trans, y_trans, z_trans, i)
        # points move 1 by 1? not all together
        
        if show_room == True:
            ax.plot3D(room_x, room_y, room_z, 'k.')

        plot = ax.plot3D(new_head_x, new_head_y, new_head_z, 'b.')
        #new_head_x, new_head_y, new_head_z = head_x, head_y, head_z
        plt.pause(0.0001)

        print('Loop', i)
        ax.clear()
        
# load all data into single array
affine_data = load_affine_data()

#plot_movement(show_trans_coors=True)
#plot_head(show_trans=False, show_head=True, show_room=True)
#show_animation(affine_data, show_room=True, small_room=True, start_frame=600)
show_old_animation(affine_data, show_room=True, small_room=True, start_frame=0)

print('Fin')