import torch

def db_method(channels, M):
    """
    DB方法的主函数
    这段实现了一种基于距离的分割方法，主要用于多通道图像的分类和边界标记。
    :param channels: 输入的多个通道，形状为 (batch_size, n, H, W)
    :param M: 每个通道中的类别数
    :return: 结果分类后的单一通道，形状为 (batch_size, H, W)
    作用：这是算法的主函数，负责调用其他函数，完成整体的边界合并、区域标记、分类等工作。
    输入：channels是输入的多个通道，形状为 (batch_size, n, H, W)；M是每个通道中的类别数。
    输出：返回分类后的单一通道，形状为 (batch_size, H, W)。
    """
    batch_size, n, H, W = channels.shape  # 提取输入张量的形状

    # 创建一个全零的布尔张量，用于存放合并后的边界
    B_new = torch.zeros((batch_size, H, W), dtype=torch.bool)

    # 将各个通道的边界合并到一起
    for channel in channels.permute(1, 0, 2, 3):  # 调整维度为 (n, batch_size, H, W)
        for m in range(M):
            B_new |= (channel == m)  # 将每个通道的边界添加到B_new

    results = []  # 存储每个样本的结果
    for b in range(batch_size):
        # 标记区域
        regions = label_regions(B_new[b])
        # 找到每个区域的最优点
        optimal_points = find_optimal_points(regions)
        # 对最优点进行排序
        optimal_points.sort(key=lambda p: tuple(p))
        # 计算每个区域包含的不同类别数和类别集合
        num, pc = calculate_num_pc(optimal_points, channels[:, b, :, :], M)
        # 根据距离计算方法对区域进行分类
        result = db_calculate_method(optimal_points, num, pc, channels[:, b, :, :])
        results.append(result)
    
    return torch.stack(results)  # 将结果堆叠成一个张量

def label_regions(boundary):
    """
    根据边界标记区域
    :param boundary: 边界图，形状为 (H, W)
    :return: 标记后的区域，形状为 (H, W)
    作用：标记边界区域，返回唯一标记和标记后的区域。
    原因：标记区域可以将图像分割成多个独立的区域，便于对每个区域进行独立处理。
    """
    # 标记边界区域，返回唯一标记和标记后的区域
    labeled, num_features = torch.unique(boundary, return_inverse=True)
    labeled = labeled.to(dtype=torch.int64)  # 确保标签是整数类型
    return labeled.view(boundary.shape)  # 确保形状匹配

def find_optimal_points(regions):
    """
    找到每个区域的最优点
    :param regions: 标记后的区域图
    :return: 每个区域的最优点
    作用：找到每个区域的最优点，用于后续的分类。
    原因：选择最佳点作为代表点，便于后续的距离计算和分类操作。
    """
    optimal_points = []
    unique_regions = torch.unique(regions)
    for region in unique_regions:
        points = torch.nonzero(regions == region, as_tuple=False)
        optimal_point = points[0]  # 默认选择第一个点作为最优点
        for point in points:
            for i in range(len(point)):
                if point[i] < optimal_point[i]:
                    optimal_point = point
                    break
                elif point[i] > optimal_point[i]:
                    break
        optimal_points.append(tuple(optimal_point.tolist()))
    return optimal_points

def calculate_num_pc(optimal_points, channels, M):
    """
    计算每个区域包含的不同类别数和类别集合
    :param optimal_points: 每个区域的最优点
    :param channels: 多通道图像
    :param M: 类别数
    :return: 每个区域包含的不同类别数和类别集合
    作用：计算每个区域包含的不同类别数和类别集合。
    原因：确定每个区域的类别数和类别集合，为后续的分类操作提供依据。
    """
    num = []
    pc = []
    for point in optimal_points:
        categories = set()
        for channel in channels:
            categories.add(channel[tuple(point)])
        num.append(len(categories))
        pc.append(categories)
    return num, pc

def db_calculate_method(optimal_points, num, pc, channels):
    """
    DB计算方法，根据距离计算区域分类结果
    :param optimal_points: 每个区域的最优点
    :param num: 每个区域包含的不同类别数
    :param pc: 每个区域的类别集合
    :param channels: 多通道图像
    :return: 分类结果
    作用：根据距离计算方法对区域进行分类。
    原因：利用距离计算方法，确定每个区域的分类结果。
    """
    result = torch.zeros(channels[0].shape, dtype=torch.long)
    for i, point in enumerate(optimal_points):
        if num[i] == 1:
            result[tuple(point)] = list(pc[i])[0]
        else:
            distances = calculate_distances(tuple(point), pc[i], channels)
            result[tuple(point)] = min(distances, key=distances.get)
    return result

def calculate_distances(point, categories, channels):
    """
    计算点到各个类别边界的距离
    :param point: 点的坐标
    :param categories: 类别集合
    :param channels: 多通道图像
    :return: 点到各个类别边界的距离
    作用：计算点到各个类别边界的距离，用于分类决策。
    原因：通过距离计算，判断点更接近哪个类别的边界，确定其分类。
    """
    distances = {}
    for category in categories:
        distance = 0
        for i, channel in enumerate(channels):
            if channel[point] != category:
                boundary = find_boundary(channel, category)
                distance += torch.dist(torch.tensor(point, dtype=torch.float32),
                                       torch.tensor(boundary, dtype=torch.float32))
        distances[category] = distance
    return distances

def find_boundary(channel, category):
    """
    找到某个类别的边界
    :param channel: 通道图像
    :param category: 类别
    :return: 类别边界
    作用：找到某个类别的边界点。
    原因：边界点用于距离计算，确定点到边界的距离。
    """
    boundaries = torch.nonzero(channel == category, as_tuple=False)
    return boundaries[0]  # 返回第一个边界点
