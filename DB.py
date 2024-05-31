import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.segmentation import find_boundaries

# 示例数据：三个通道的分类结果
layer_r = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 2]])
layer_g = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 3]])
layer_b = np.array([[1, 1, 1], [1, 2, 2], [3, 3, 3]])

# 计算每个通道的边界线
def compute_boundary(layer):
    return find_boundaries(layer, mode='inner')


# 对三个通道分别计算边界线
boundary_r = compute_boundary(layer_r)
boundary_g = compute_boundary(layer_g)
boundary_b = compute_boundary(layer_b)

# 合并边界线，形成一个新的通道（combined_boundary），表示所有层的分界线的并集
combined_boundary = np.maximum(boundary_r, np.maximum(boundary_g, boundary_b))

print("合并后的边界线：")
print(combined_boundary)

# 标记合并后的边界线形成的区域
# labeled_regions：标记每个区域的ID
# num_regions：区域的数量
labeled_regions, num_regions = label(combined_boundary == 0)
print("标记后的区域：")
print(labeled_regions)
print("区域数量：", num_regions)
# 计算每个点到各个类别边界的距离
# layers：包含三个通道的分类结果
def compute_distances(layers):
    distances = []
    for layer in layers:
        unique_labels = np.unique(layer)  # 找到每个通道中的唯一类别标签
        # 为每个类别标签计算距离变换
        distance_map = {label: distance_transform_edt(layer != label) for label in unique_labels}
        distances.append(distance_map)
    return distances
layers = [layer_r, layer_g, layer_b]
# distances：一个列表，每个元素是一个字典，字典键是类别标签，值是到该类别边界的距离矩阵
distances = compute_distances(layers)
# 确定每个区域的分类，只考虑pc=2和pc=3的情况
def determine_class(layers, distances, labeled_regions, num_regions):
    result_layer = np.zeros_like(labeled_regions)  # 用于存储最终分类结果的矩阵
    for region_id in range(1, num_regions + 1):
        # 获取当前区域的掩码
        region_mask = (labeled_regions == region_id)
        # 获取区域内所有点的坐标
        i_indices, j_indices = np.where(region_mask)
        # 获取当前区域内所有点在各通道上的类别组合
        point_classes = [tuple(layers[k][i_indices, j_indices]) for k in range(len(layers))]
        # 确定当前区域内所有点在各通道上的唯一类别组合
        unique_classes = list(set(point_classes[0]).union(*point_classes[1:]))
        if len(unique_classes) == 1:
            # 如果区域内所有点在所有通道上都属于同一个类别，直接赋值
            result_layer[region_mask] = unique_classes[0][0]
        elif len(unique_classes) == 2:
            # 处理pc=2的情况
            class_pairs = [(unique_classes[0][0], unique_classes[0][1], unique_classes[1][0]),
                           (unique_classes[1][0], unique_classes[1][1], unique_classes[0][0])]
            min_distance = float('inf')
            best_class = None
            for cls in class_pairs:
                d1 = sum(distances[2][cls[2]][region_mask])
                d2 = sum(distances[0][cls[0]][region_mask]) + sum(distances[1][cls[1]][region_mask])
                if d1 < d2:
                    if d1 < min_distance:
                        min_distance = d1
                        best_class = cls[0]
                else:
                    if d2 < min_distance:
                        min_distance = d2
                        best_class = cls[2]
            result_layer[region_mask] = best_class
        elif len(unique_classes) == 3:
            # 处理pc=3的情况
            # 确定当前区域内所有点在各通道上的类别组合
            class_combinations = [(unique_classes[0][0], unique_classes[1][0], unique_classes[2][0])]
            min_distance = float('inf')
            best_class = None
            for cls in class_combinations:
                # d1: 将区域点分配到第一个类别的总距离
                d1 = sum(distances[1][cls[0]][region_mask]) + sum(distances[2][cls[0]][region_mask])
                # d2: 将区域点分配到第二个类别的总距离
                d2 = sum(distances[0][cls[1]][region_mask]) + sum(distances[2][cls[1]][region_mask])
                # d3: 将区域点分配到第三个类别的总距离
                d3 = sum(distances[0][cls[2]][region_mask]) + sum(distances[1][cls[2]][region_mask])
                # distances_sum: 每个类别组合的总距离列表
                distances_sum = [d1, d2, d3]
                # min_idx: 最小总距离的索引
                min_idx = np.argmin(distances_sum)

                # 如果当前类别组合的总距离小于之前的最小距离，更新最小距离和最佳类别
                if distances_sum[min_idx] < min_distance:
                    min_distance = distances_sum[min_idx]
                    best_class = cls[min_idx]
            # 将最佳类别分配给当前区域
            result_layer[region_mask] = best_class
    return result_layer
result_layer = determine_class(layers, distances, labeled_regions, num_regions)
print("分类结果：")
print(result_layer)
