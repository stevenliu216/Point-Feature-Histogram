import numpy
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_pc(filename):
    """Load a csv PC.

    Loads a point cloud from a csv file.

    inputs:
        filename - a string containing the files name.
    outputs:
        pc - a list of 3 by 1 numpy matrices that represent the points.

    """
    pc = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            pc.append(numpy.matrix([float(x) for x in row]).T)

    return pc

def load_pc_np(filename):
    """Load a numpy PC.

    Loads a point cloud from a npy file.

    inputs:
        filename - a string containing the files name.
    outputs:
        pc - a list of 3 by 1 numpy matrices that represent the points.

    """
    pc_source = numpy.load(filename)
    numPoints = len(pc_source)
    pointCloudSource = []
    row = []
    for i in range(numPoints):
        for j in range(3):
            row.append(pc_source[i][j])
        pointCloudSource.append(numpy.matrix([float(x) for x in row]).T)
        row = []

    return pointCloudSource

def view_pc(pcs, fig=None, color='b', marker='o'):
    """Visualize a pc.

    inputs:
        pc - a list of numpy 3 x 1 matrices that represent the points.
        color - specifies the color of each point cloud.
            if a single value all point clouds will have that color.
            if an array of the same length as pcs each pc will be the color corresponding to the
            element of color.
        marker - specifies the marker of each point cloud.
            if a single value all point clouds will have that marker.
            if an array of the same length as pcs each pc will be the marker corresponding to the
            element of marker.
    outputs:
        fig - the pyplot figure that the point clouds are plotted on

    """
    # Construct the color and marker arrays
    if hasattr(color, '__iter__'):
        if len(color) != len(pcs):
            raise Exception('color is not the same length as pcs')
    else:
        color = [color] * len(pcs)

    if hasattr(marker, '__iter__'):
        if len(marker) != len(pcs):
            raise Exception('marker is not the same length as pcs')
    else:
        marker = [marker] * len(pcs)

    # Start plt in interactive mode
    ax = []
    if fig == None:
        plt.ion()
        # Make a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()
        ax.hold()

    # Draw each point cloud
    for pc, c, m in zip(pcs, color, marker):
        x = []
        y = []
        z = []
        for pt in pc:
            x.append(pt[0, 0])
            y.append(pt[1, 0])
            z.append(pt[2, 0])

        ax.scatter(x, y, z, color=c, marker=m)

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.hold()
    # Update the figure
    plt.show()

    # Return a handle to the figure so the user can make adjustments
    return fig


def draw_plane(fig, normal, pt, color=(0.1, 0.2, 0.5, 0.3), length=[-1, 1], width=[-1, 1]):
    """Draws a plane on fig.

    inputs:
        fig - the matplotlib object to plot on.
        normal - a 3 x 1 numpy matrix representing the normal of the plane.
        pt - a 3 x 1 numpy matrix representing a point on the plane.
        color - the color of the plane specified as in matplotlib
        width - the width of the plane specified as [min, max]
        length - the length of the plane specified as [min, max]
    outputs:
        fig - the matplotlib object to plot on.

    """
    # Calculate d in ax + by + cz + d = 0
    d = -pt.T * normal

    # Calculate points on the surface
    x = 0
    y = 0
    z = 0
    if normal[2, 0] != 0:
        x, y = numpy.meshgrid(numpy.linspace(length[0], length[1], 10),
                              numpy.linspace(width[0], width[1], 10))
        z = (-d - normal[0, 0] * x - normal[1, 0] * y) / normal[2, 0]
    elif normal[1, 0] != 0:
        x, z = numpy.meshgrid(numpy.linspace(length[0], length[1], 10),
                              numpy.linspace(width[0], width[1], 10))
        y = (-d - normal[0, 0] * x - normal[2, 0] * z) / normal[1, 0]
    elif normal[0, 0] != 0:
        y, z = numpy.meshgrid(numpy.linspace(length[0], length[1], 10),
                              numpy.linspace(width[0], width[1], 10))
        x = (-d - normal[1, 0] * y - normal[2, 0] * z) / normal[0, 0]

    # Plot the surface
    ax = fig.gca()
    ax.hold()
    ax.plot_surface(x, y, z, color=color)
    return fig


def add_noise(pc, variance,distribution='gaussian'):
    """Add Gaussian noise to pc.

    For each dimension randomly sample from a Gaussian (N(0, Variance)) and add the result
        to the dimension dimension.

    inputs:
        pc - a list of numpy 3 x 1 matrices that represent the points.
        variance - the variance of a 0 mean Gaussian to add to each point or width of the uniform distribution
        distribution - the distribution to use (gaussian or uniform)
    outputs:
        pc_out - pc with added noise.

    """
    pc_out = []

    if distribution=='gaussian':
        for pt in pc:
            pc_out.append(pt + numpy.random.normal(0, variance, (3, 1)))
    elif distribution=='uniform':
        for pt in pc:
            pc_out.append(pt + numpy.random.uniform(-variance, variance, (3, 1)))
    else:
        raise ValueError(['Unknown distribution type: ', distribution])
    return pc_out


def merge_clouds(pc1, pc2):
    """Add Gaussian noise to pc.

    Merge two point clouds

    inputs:
        pc1 - a list of numpy 3 x 1 matrices that represent one set of points.
        pc2 - a list of numpy 3 x 1 matrices that represent another set of points.
    outputs:
        pc_out - merged point cloud

    """
    pc_out = pc1
    for pt in pc2:
        pc_out.append(pt)

    return pc_out

def add_outliers(pc, multiple_of_data, variance, distribution='gaussian'):
    """Add outliers to pc.

    inputs:
        pc - a list of numpy 3 x 1 matrices that represent the points.
        multiple_of_data - how many outliers to add in terms of multiple of data. Must be an integer >= 1.
        variance - the variance of a 0 mean Gaussian to add to each point.
        distribution - the distribution to use (gaussian or uniform)
    outputs:
        pc_out - pc with added outliers.

    """
    pc_out = pc
    for i in range(0,multiple_of_data):
        pc_outliers = add_noise(pc_out, variance,distribution)
        pc_out = merge_clouds(pc_out,pc_outliers)
    return pc_out

def add_outliers_centroid(pc, num_outliers, variance, distribution='gaussian'):
    """Add outliers to pc (reference to centroid).


    inputs:
        pc - a list of numpy 3 x 1 matrices that represent the points.
        num_outliers - how many outliers to add
        variance - the variance of a 0 mean Gaussian to add to each point.
        distribution - the distribution to use (gaussian or uniform)
    outputs:
        pc_out - pc with added outliers.

    """
    centroid = numpy.zeros((3, 1))
    for pt in pc:
        centroid = centroid + pt
    centroid = centroid/len(pc)

    newpoints = []
    for i in range(0,num_outliers):
        newpoints.append(numpy.matrix(centroid))

    return merge_clouds(pc, add_noise(newpoints,variance,distribution))

def transform_cloud(pc,tx,ty,tz,roll,pitch,yaw):
    # Generate transform
    c = numpy.cos(roll)
    s = numpy.sin(roll)
    Rx = numpy.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    c = numpy.cos(pitch)
    s = numpy.sin(pitch)
    Ry = numpy.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c = numpy.cos(yaw)
    s = numpy.sin(yaw)
    Rz = numpy.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rx * Ry * Rz
    t = numpy.matrix([[tx, ty, tz]]).T
    H = numpy.matrix(numpy.identity(4))
    H[0:3, 0:3] = R
    H[0:3, 3] = t
    print('Transform is:\n', H)
    pc_out = []
    for pt in pc:
        hpoint = numpy.matrix([pt[0,0],pt[1,0],pt[2,0],1]).T
        hpoint = H * hpoint
        pc_out.append(hpoint[0:3,0])
    return pc_out

def convert_pc_to_matrix(pc):
    """Coverts a point cloud to a numpy matrix.

    Inputs:
        pc - a list of 3 by 1 numpy matrices.
    outputs:
        numpy_pc - a 3 by n numpy matrix where each column is a point.

    """
    numpy_pc = numpy.matrix(numpy.zeros((3, len(pc))))

    for index, pt in enumerate(pc):
        numpy_pc[0:3, index] = pt

    return numpy_pc

def convert_matrix_to_pc(numpy_pc):
    """Coverts a numpy matrix to a point cloud (useful for plotting).

    Inputs:
        numpy_pc - a 3 by n numpy matrix where each column is a point.
    outputs:
        pc - a list of 3 by 1 numpy matrices.

    """
    pc = []

    for i in range(0,numpy_pc.shape[1]):
        pc.append((numpy_pc[0:3,i]))

    return pc
