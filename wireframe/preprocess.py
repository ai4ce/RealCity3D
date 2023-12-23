import os
import numpy as np

def quantize_verts(verts):
    """Quantizes vertices and outputs integers with specified n_bits."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2**8 - 1
    verts_quantize = (
        (verts - min_range) * range_quantize / (max_range - min_range))
    return verts_quantize.astype(int)

cnt = 0
dir_name = "Brooklyn"
for file_name in os.listdir("Datasets/"+dir_name):
    with open("Datasets/" + dir_name +"/"+file_name, "r") as lines:
        vertices = []   # each: [z,y,x] float
        faces = []      # each: [v1, v2, ...]   int
        while 1:
            line = lines.readline()
            if line == "":
                break
            line = line.split(" ")
            if line[0] != "v" and line[0] != "f":
                continue
            if line[0] == "v":
                vertices.append(list(map(float, [line[3], line[2], line[1]])))
            if line[0] == "f":
                faces.append(list(map(int, line[1:])))
    vertices = np.array(vertices)   # [V, 3]
    if (vertices.shape[0] < 20):
        continue
    scale = vertices.max(axis=0)-vertices.min(axis=0)    # 极差 [3,]
    scale_max = scale.max()
    medium = 0.5* (vertices.max(axis=0) + vertices.min(axis=0)) # boundingbox中心店
        
    # normalize from medium with max length, then in [-0.5, 0.5]
    normed_vertices = (vertices - medium) / scale_max

    # quantize, then all coordinates in [0, 255]
    quan_vertices = quantize_verts(normed_vertices) # [V, 3]

    # merge 相同点
    num_v = quan_vertices.shape[0]
    num_diff_v = 0
    diff_v = []
    new_index = {}
    for i in range(num_v):
        not_in = True
        for itv in diff_v:
            if (quan_vertices[i,:]==itv).all():
                not_in = False
                break
        if not_in == True:
            num_diff_v += 1
            diff_v.append(quan_vertices[i,:])
        new_index[i+1] = num_diff_v
    vertices_merged = np.array(diff_v)

    for f_i in range(len(faces)):
        faces[f_i] = [new_index[x] for x in faces[f_i]] # 每行 f 换新index
        faces[f_i] = [i for n, i in enumerate(faces[f_i]) if i not in faces[f_i][:n]]
    #删除重复f行
    faces_merged = list(set(tuple(sub) for sub in faces)) # [(), (), ...]


    # sort (vertices & faces)
    num_v = vertices_merged.shape[0]
    vertices = np.concatenate([vertices_merged, 
                       np.array(range(num_v)).reshape(num_v,1)+1], axis=1)
    vertices = vertices.tolist()
    vertices.sort()

    new_index = {}
    for i in range(num_v):
        new_index[vertices[i][-1]] = i+1
    
    for f_i in range(len(faces_merged)):
        faces[f_i] = [new_index[x] for x in faces[f_i]]
        ind = faces[f_i].index(min(faces[f_i]))
        tmp = faces[f_i][ind:] + faces[f_i][:ind]
        faces[f_i] = tmp
    faces.sort(key= lambda l: sorted(l))
    for f_i in range(len(faces)):
        faces[f_i] = list(map(str, faces[f_i]))
    
    # writing into new 123.obj
    os.makedirs("Datasets/" + dir_name +"_processed", exist_ok=True)
    with open("Datasets/" + dir_name +"_processed/"+
              str(cnt) + ".obj", "w") as file:
        for v in vertices:
            file.write(f"v {v[2]} {v[1]} {v[0]}\n")
        for f in faces:
            file.write("f " + " ".join(f) + "\n")
    #print(cnt, file_name)
    print(cnt)
    cnt += 1