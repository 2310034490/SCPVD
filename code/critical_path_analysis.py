import networkx as nx
import numpy as np


class CriticalPathAnalyzer:
    """
    基于关键节点和中心性分析的关键路径生成器
    
    核心创新点：
    1. 基于预定义规则发现潜在漏洞行
    2. 结合三种节点中心性分析（度中心性、接近中心性、介数中心性）定义关键节点
    3. 通过贪心算法高效构建覆盖最多关键节点的关键路径
    """

    def __init__(self):
        self.vulnerability_lines = set()  # 漏洞相关行集合

    def set_vulnerability_lines(self, vuln_lines):
        """
        设置与漏洞相关的代码行，这些行应该被保留在关键路径中
        
        Args:
            vuln_lines: 漏洞相关行号的集合
        """
        self.vulnerability_lines = set(vuln_lines)

    def calculate_centrality_measures(self, G):
        """
        计算图中节点的中心性度量，包括度中心性、接近中心性和介数中心性
        
        Args:
            G: 要分析的图（networkx.DiGraph）
            
        Returns:
            centrality_scores: 包含各种中心性度量的字典
        """
        centrality_scores = {}
        
        # 计算度中心性 (Degree Centrality)
        # 度中心性衡量节点的连接数量，表示节点的直接影响力
        in_degree_centrality = nx.in_degree_centrality(G)  # 入度中心性
        out_degree_centrality = nx.out_degree_centrality(G)  # 出度中心性
        degree_centrality = {}
        for node in G.nodes():
            # 综合考虑入度和出度中心性
            degree_centrality[node] = (in_degree_centrality.get(node, 0) + out_degree_centrality.get(node, 0)) / 2
        centrality_scores['degree'] = degree_centrality
        
        # 计算接近中心性 (Closeness Centrality)
        # 接近中心性衡量节点到其他所有节点的平均最短路径长度的倒数，表示信息传播的效率
        try:
            # 对于有向图，计算接近中心性可能会因为不连通而失败
            closeness_centrality = nx.closeness_centrality(G)
            centrality_scores['closeness'] = closeness_centrality
        except:
            # 如果计算失败，使用默认值
            centrality_scores['closeness'] = {node: 0.0 for node in G.nodes()}
        
        # 计算介数中心性 (Betweenness Centrality)
        # 介数中心性衡量节点在网络中作为桥梁的重要性，表示控制信息流的能力
        try:
            # 介数中心性计算可能很耗时，对于大图可以使用近似算法
            if len(G.nodes()) > 500:
                # 对于大图使用近似算法
                betweenness_centrality = nx.approximation.betweenness_centrality(G)
            else:
                # 对于小图使用精确算法
                betweenness_centrality = nx.betweenness_centrality(G)
            centrality_scores['betweenness'] = betweenness_centrality
        except:
            # 如果计算失败，使用默认值
            centrality_scores['betweenness'] = {node: 0.0 for node in G.nodes()}
        
        return centrality_scores
    
    def identify_critical_nodes_by_centrality(self, G, first_line, exit_lines, vulnerability_lines=None, threshold=0.7):
        """
        基于中心性度量识别关键节点
        
        Args:
            G: 要分析的图（networkx.DiGraph）
            first_line: 入口节点
            exit_lines: 出口节点列表
            vulnerability_lines: 漏洞相关行列表
            threshold: 中心性阈值，高于此值的节点被视为关键节点
            
        Returns:
            critical_nodes: 关键节点集合
        """
        # 初始关键节点：入口、出口和漏洞相关行
        critical_nodes = {first_line} | set(exit_lines)
        if vulnerability_lines:
            critical_nodes |= set([v for v in vulnerability_lines if v in G.nodes()])

        centrality_selected_nodes = set()
        
        # 计算中心性度量
        centrality_scores = self.calculate_centrality_measures(G)
        
        # 归一化中心性分数
        normalized_scores = {}
        for node in G.nodes():
            # 综合考虑各种中心性度量
            degree_score = centrality_scores['degree'].get(node, 0)
            closeness_score = centrality_scores['closeness'].get(node, 0)
            betweenness_score = centrality_scores['betweenness'].get(node, 0)
            
            # 加权平均，可以根据需要调整权重
            # 增加度中心性的权重，使具有多入度/出度的节点更容易被识别为关键节点
            combined_score = 0.5 * degree_score + 0.2 * closeness_score + 0.3 * betweenness_score
            normalized_scores[node] = combined_score
        
        # 根据图的大小动态调整阈值
        node_count = len(G.nodes())
        dynamic_threshold = threshold
        if node_count > 100:
            # 对于大图，降低阈值以包含更多关键节点
            dynamic_threshold = threshold * 0.8
        elif node_count < 20:
            # 对于小图，提高阈值以减少关键节点数量
            dynamic_threshold = threshold * 1.2
        
        # 如果有足够的节点，使用百分位数作为阈值
        if node_count > 10:
            scores = list(normalized_scores.values())
            percentile_threshold = np.percentile(scores, dynamic_threshold * 100)
            
            # 添加高于阈值的节点为关键节点
            for node, score in normalized_scores.items():
                if score >= percentile_threshold:
                    critical_nodes.add(node)
                    centrality_selected_nodes.add(node)
                # 特别处理：确保具有多入度或多出度的节点被识别为关键节点
                elif G.in_degree(node) > 1 and G.out_degree(node) > 1:
                    critical_nodes.add(node)
                    centrality_selected_nodes.add(node)
        else:
            # 对于小图，直接使用排序后的前N个节点
            sorted_nodes = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
            top_n = max(1, int(node_count * dynamic_threshold))
            for node, _ in sorted_nodes[:top_n]:
                critical_nodes.add(node)
                centrality_selected_nodes.add(node)

        self.last_centrality_selected_nodes = centrality_selected_nodes
        return critical_nodes
    
    def generate_critical_node_path(self, G, first_line, exit_lines, critical_nodes):
        """
        生成包含所有关键节点的路径，优先覆盖最多的关键节点
        使用贪心算法高效构建关键路径
        
        Args:
            G: 要分析的图（networkx.DiGraph）
            first_line: 入口节点
            exit_lines: 出口节点列表
            critical_nodes: 关键节点集合
            
        Returns:
            critical_path: 包含尽可能多关键节点的路径
        """
        # 确保入口、出口和漏洞相关行都被包含在关键节点中
        if first_line not in critical_nodes:
            critical_nodes.add(first_line)
        
        for exit_line in exit_lines:
            if exit_line in G.nodes() and exit_line not in critical_nodes:
                critical_nodes.add(exit_line)
        
        # 从入口节点开始
        critical_path = [first_line]
        visited = {first_line}
        current = first_line
        remaining_critical = set(critical_nodes) - {first_line}
        
        # 贪心算法：每次选择能够覆盖最多未访问关键节点的路径
        while remaining_critical and len(critical_path) < len(G.nodes()):
            best_path = None
            best_score = -1
            best_target = None
            
            # 尝试找到一条能覆盖最多关键节点的路径
            for target in remaining_critical:
                try:
                    if nx.has_path(G, current, target):
                        # 找到最短路径
                        path = nx.shortest_path(G, current, target)
                        
                        # 计算路径中包含的未访问关键节点数量
                        critical_covered = set(path) & remaining_critical
                        
                        # 评分：覆盖的关键节点数量减去路径长度的惩罚
                        # 优先选择覆盖更多关键节点的路径，其次是更短的路径
                        score = len(critical_covered) - 0.01 * len(path)
                        
                        if score > best_score:
                            best_score = score
                            best_path = path
                            best_target = target
                except nx.NetworkXNoPath:
                    continue
            
            if best_path:
                # 将路径中间节点添加到结果中
                for node in best_path[1:]:  # 排除当前节点
                    if node not in visited:
                        critical_path.append(node)
                        visited.add(node)
                        if node in remaining_critical:
                            remaining_critical.remove(node)
                
                current = best_path[-1]  # 更新当前位置为路径终点
            else:
                # 如果无法到达任何剩余的关键节点，尝试连接到出口
                exit_connected = False
                for exit_line in exit_lines:
                    if exit_line not in visited:
                        try:
                            if nx.has_path(G, current, exit_line):
                                path = nx.shortest_path(G, current, exit_line)
                                for node in path[1:]:  # 排除当前节点
                                    if node not in visited:
                                        critical_path.append(node)
                                        visited.add(node)
                                        if node in remaining_critical:
                                            remaining_critical.remove(node)
                                exit_connected = True
                                break
                        except nx.NetworkXNoPath:
                            continue
                
                # 如果无法连接到任何出口，退出循环
                if not exit_connected:
                    break
        
        # 如果还有未访问的关键节点，记录警告信息
        # if remaining_critical:
            # print(f"警告：无法在路径中包含所有关键节点，剩余 {len(remaining_critical)} 个未访问的关键节点")
        
        return critical_path

    def analyze_and_generate_path(self, G, first_line, exit_lines, vulnerability_lines=None, threshold=0.7):
        """
        完整的关键路径分析和生成流程
        
        Args:
            G: 要分析的图（networkx.DiGraph）
            first_line: 入口节点
            exit_lines: 出口节点列表
            vulnerability_lines: 漏洞相关行列表
            threshold: 中心性阈值
            
        Returns:
            critical_path: 关键路径
            critical_nodes: 识别出的关键节点集合
        """
        # 设置漏洞相关行
        if vulnerability_lines:
            self.set_vulnerability_lines(vulnerability_lines)
        
        # 识别关键节点
        critical_nodes = self.identify_critical_nodes_by_centrality(
            G, first_line, exit_lines, vulnerability_lines, threshold
        )
        
        # 生成关键路径
        critical_path = self.generate_critical_node_path(G, first_line, exit_lines, critical_nodes)
        
        return critical_path, critical_nodes
