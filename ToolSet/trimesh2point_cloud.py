import os
import argparse
import numpy as np
import trimesh
import open3d as o3d

import multiprocessing
from timeit import default_timer as timer
from datetime import timedelta

def tri2pts(in_path,num_points,out_dir):
    trimesh.util.attach_to_log()
    mesh = trimesh.load(in_path)
    points = trimesh.sample.sample_surface(mesh,4096)

    point_mat = np.array(points[0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_mat)
    o3d.io.write_point_cloud(out_dir+in_path.split('/')[-1][:-4]+".ply", pcd)
    return


def main():
    parser = argparse.ArgumentParser(description='Point CLoud Sampling Tool')
    parser.add_argument('-i', '--in_dir', type=str, default="./tri_mesh/", help='input directory')
    parser.add_argument('-o', '--out_dir', type=str, default="./point_clouds/", help='output directory')
    parser.add_argument('-t', '--num_proc', type=int, default=24, help='Number of threads')
    parser.add_argument('-p', '--num_points', type=int, default=4096, help='Number of points')

    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    num_points = args.num_points
    num_proc = args.num_proc

    list_of_files = os.listdir(in_dir)
    obj_files = [x for x in list_of_files if x[-4:] == ".obj"]
    
    max_iters = int(len(obj_files) / num_proc) + 1

    for iter in range(max_iters):       
        jobs = []
        for i in range(num_proc):
            curr_id = iter * num_proc + i
            if curr_id >= len(obj_files):
                break
            f = obj_files[curr_id]
            in_path = os.path.join(in_dir, f)
            
            p = multiprocessing.Process(target=tri2pts, args=(in_path,num_points,out_dir,))
            jobs.append(p)
            p.start()
            print("P {} started!".format(i))
    return
    

if __name__ == '__main__':
    main()
