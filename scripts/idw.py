import os
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

class Vector3:
    def __init__(self, x, y, z, nodeID):
        self.x = x
        self.y = y
        self.z = z
        self.nodeID = nodeID

    def to_tuple(self):
        return (self.x, self.y, self.z)


class NodeData:
    def __init__(self, nodeID=0, origX=0.0, origY=0.0, origZ=0.0, value=0.0):
        self.nodeID = nodeID
        self.origX = origX
        self.origY = origY
        self.origZ = origZ
        self.value = value


class DataParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fluent_data = []

    def parse_file(self):
        start_parse = False
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件 {self.file_path} 不存在。")
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if 'nodenumber' in line:
                    start_parse = True
                    continue
                if start_parse and line:
                    parts = line.split()
                    if len(parts) == 9:
                        try:
                            node = NodeData(int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                                            float(parts[8]))
                            self.fluent_data.append(node)
                        except ValueError as e:
                            print(f"数据转换错误在行 {line_number}: {e}")


def read_positions(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在。")
    positions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            parts = line.strip().split()
            if len(parts) >= 4:
                nodeID, x, y, z = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                positions.append(Vector3(x, y, z, nodeID))
    return positions


def inverse_distance_weighted_interpolation(position, nodes, values, k=6):
    if len(nodes) < k:
        k = len(nodes)
    kdtree = cKDTree(nodes)
    distances, indices = kdtree.query(position.to_tuple(), k=k)
    if isinstance(distances, float):  # 当k=1时返回的是单个值而不是数组
        distances = [distances]
        indices = [indices]

    numerator = 0.0
    denominator = 0.0
    for dist, idx in zip(distances, indices):
        if dist < 1e-6:
            return values[idx] if values[idx] >= 0.001 else 0  # 如果非常接近某个节点，返回该节点的值（如果大于0.001）
        weight = 1.0 / dist
        numerator += weight * values[idx]
        denominator += weight
    interpolated_value = numerator / denominator if denominator != 0 else 0
    return interpolated_value if interpolated_value >= 0.001 else 0  # 如果插值结果小于0.001，则取值为0


def interpolate_values(file_path, positions, k=4):
    print(f"正在处理文件: {file_path}")
    node_parser = DataParser(file_path)
    node_parser.parse_file()
    nodes = [(node.origX, node.origY, node.origZ) for node in node_parser.fluent_data]
    values = [node.value for node in node_parser.fluent_data]

    results = []
    for pos in positions:
        interpolated_value = inverse_distance_weighted_interpolation(pos, nodes, values, k)
        results.append((pos.nodeID, pos.x, pos.y, pos.z, interpolated_value))
    return results


def save_results_to_excel(results, file_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame(results,
                      columns=['Node ID', 'X Coordinate', 'Y Coordinate', 'Z Coordinate', 'Interpolated Value'])
    file_base_name = os.path.splitext(os.path.basename(file_name))[0]
    file_path = os.path.join(save_path, f'{file_base_name}_results.xlsx')
    df.to_excel(file_path, index=False)
    print(f"结果已保存至: {file_path}")


def process_and_save(file_path, positions, save_path):
    results = interpolate_values(file_path, positions)
    save_results_to_excel(results, file_path, save_path)


def main(data_path, position_file, save_path):
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if
             f.startswith('VandCh4-') and f.endswith('.txt')]
    files.sort()
    positions = read_positions(position_file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_and_save, file, positions, save_path) for file in files]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    data_path = '原始数据路径 data\\initial'
    position_file = '空间节点文件\\data\\Initial\\shiwu3m_nodes.txt'
    save_path = '保存\\data\\V'
    main(data_path, position_file, save_path)
