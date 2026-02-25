import math
import os, sys, argparse
import inspect
import json
import pdb
import numpy as np
import scannet_utils

def read_aggregation(filename):
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file):
    """ points   XYZ RGB (RGB in 0-255),
    semantic label  nyu40 ids,
    instance label 1-#instance,
    box  (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    label_map = scannet_utils.read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')
 # 数据增强，9维度
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb_normal(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    axis_align_matrix = None
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    #aligned_vertices，矩阵相乘，坐标变换 x y z 变换到ScanNet 的统一坐标系
    if axis_align_matrix != None:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        pts = np.ones((mesh_vertices.shape[0], 4))
        pts[:,0:3] = mesh_vertices[:,0:3]
        pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
        aligned_vertices = np.copy(mesh_vertices)
        aligned_vertices[:,0:3] = pts[:,0:3]
    else:
        print("No axis alignment matrix found")
        aligned_vertices = mesh_vertices
    # seg_to_verts 片段编号对应片段列表索引号的映射
    if os.path.isfile(agg_file):
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)
    # 片段列表索引号到label id的映射，list
        # 某个标签有很多seg,这些seg在场景文件中的排列索引和点云数据一样，所以可以通过这些seg索引筛选出对应的点云坐标
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
      #  片段列表索引号到object_id的映射，list  第一个label_id label_ids[verts][0]
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        num_instances = len(np.unique(list(object_id_to_segs.keys())))
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]
        
        instance_bboxes = np.zeros((num_instances,8)) # also include object id
        aligned_instance_bboxes = np.zeros((num_instances,8)) # also include object id
        for obj_id in object_id_to_segs:
            label_id = object_id_to_label_id[obj_id]
        #seges.json 的segIndices 顺序和加载获得的点云顺序一样
           # instance_ids==obj_id 为true才取
           # 选出某个物体的所有点云坐标
            obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
            if len(obj_pc) == 0: continue
            # 盒子的中心点，长宽高
        
            xmin = np.min(obj_pc[:,0])
            ymin = np.min(obj_pc[:,1])
            zmin = np.min(obj_pc[:,2])
            xmax = np.max(obj_pc[:,0])
            ymax = np.max(obj_pc[:,1])
            zmax = np.max(obj_pc[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, obj_id-1]) # also include object id
 
            instance_bboxes[obj_id-1,:] = bbox 

            
            obj_pc = aligned_vertices[instance_ids==obj_id, 0:3]
            if len(obj_pc) == 0: continue
    
            xmin = np.min(obj_pc[:,0])
            ymin = np.min(obj_pc[:,1])
            zmin = np.min(obj_pc[:,2])
            xmax = np.max(obj_pc[:,0])
            ymax = np.max(obj_pc[:,1])
            zmax = np.max(obj_pc[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, obj_id-1]) # also include object id
            
            aligned_instance_bboxes[obj_id-1,:] = bbox 
    else:
        # use zero as placeholders for the test scene
        print("use placeholders")
        num_verts = mesh_vertices.shape[0]
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        instance_bboxes = np.zeros((1, 8)) # also include object id
        aligned_instance_bboxes = np.zeros((1, 8)) # also include object id


    return mesh_vertices, aligned_vertices, label_ids, instance_ids, instance_bboxes, aligned_instance_bboxes
