import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

def read_obj_file(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(vertex_index.split('/')[0]) - 1 for vertex_index in line.strip().split()[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)

def extract_slice_polygons(vertices, faces, z_value):
    polygons = []

    for face in faces:
        # Check if any pair of vertices of the face has the same z value (within a small tolerance)
        vertex_indices = face
        z_values = [vertices[vertex_index][2] for vertex_index in vertex_indices]
        if any(np.isclose(z, z_value) for z in z_values):
            polygons.append(face)

    return polygons

def is_single_connected(polygons):
    # Check if the polygons form a single connected component
    return len(polygons) == 1

def main():
    obj_file_path = 'path/to/your/file.obj'
    target_z = 0.5  # Replace this with the specific z-coordinate you want

    vertices, faces = read_obj_file(obj_file_path)
    polygons = extract_slice_polygons(vertices, faces, target_z)

    if not polygons:
        print("No polygons found at the specified Z-coordinate.")
    elif not is_single_connected(polygons):
        print("Error: The extracted polygons are not single connected.")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for polygon_indices in polygons:
            polygon_vertices = vertices[polygon_indices]
            poly3d = [[polygon_vertices[vertex_index][0], polygon_vertices[vertex_index][1], polygon_vertices[vertex_index][2]] for vertex_index in range(len(polygon_vertices))]
            poly3d.append(poly3d[0])  # Close the polygon
            poly3d = np.array(poly3d)
            
            poly = Poly3DCollection([poly3d], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5)
            ax.add_collection3d(poly)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

if __name__ == "__main__":
    main()
