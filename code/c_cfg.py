import ast
import networkx as nx
from itertools import islice
from critical_path_analysis import CriticalPathAnalyzer
from tqdm import tqdm
import time
import os
import pickle
import shutil

# 全局变量，用于跟踪处理的代码片段数量
total_snippets = 0
processed_snippets = 0
progress_bar = None

total_nodes_across_dataset = 0
total_critical_nodes_across_dataset = 0
total_vuln_nodes_across_dataset = 0
total_centrality_nodes_across_dataset = 0
total_overlap_nodes_across_dataset = 0

# 缓存相关配置
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
PATH_CACHE_DIR = os.path.join(CACHE_DIR, 'paths')

# 确保缓存目录存在
os.makedirs(PATH_CACHE_DIR, exist_ok=True)


class C_CFG():
    def __init__(self):
        self.finlineno = []
        self.firstlineno = 1
        self.loopflag = 0
        self.clean_code = ''

        self.func_name = dict()
        self.G = nx.DiGraph()
        self.DG = nx.DiGraph()
        self.circle = []
        self.dece_node = []
        self.critical_path_analyzer = CriticalPathAnalyzer()  # 初始化关键路径分析器
        self.vulnerability_lines = set()  # 存储漏洞相关的代码行

        # 统计图的变量
        self.original_graph_nodes = 0
        self.original_graph_edges = 0
        
        # 缓存相关属性
        self.idx = None  # 添加idx属性用于存储代码片段的idx
        self.code_hash = None # 代码内容的哈希值
        self.path_cache_path = None # 路径缓存文件路径

    def _extract_node_code_content(self):
        """
        提取每个节点对应的代码内容，用于相似性分析。
        
        Returns:
            dict: 节点号到代码内容的映射。
        """
        code_content = {}
        
        # 将代码按行分割
        code_lines = self.clean_code.split('\n')
        
        # 为每个节点提取对应的代码行
        for node in self.G.nodes():
            try:
                # 节点号通常对应代码行号（从1开始）
                if isinstance(node, int) and 1 <= node <= len(code_lines):
                    # 获取对应行的代码内容，去除前后空白
                    line_content = code_lines[node - 1].strip()
                    code_content[node] = line_content
                else:
                    # 对于无效的节点号，使用空字符串
                    code_content[node] = ""
            except (IndexError, TypeError):
                # 处理异常情况
                code_content[node] = ""
        
        return code_content

    def _get_cache_paths(self):
        """获取缓存文件路径"""
        # 优先使用idx作为缓存文件名
        if self.idx is not None:
            # 使用代码片段的idx作为缓存文件名
            self.path_cache_path = os.path.join(PATH_CACHE_DIR, f"paths_{self.idx}.pkl")

    @staticmethod
    def clear_all_caches(self):
        """清理所有缓存文件"""
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        os.makedirs(PATH_CACHE_DIR, exist_ok=True)

    def k_shortest_paths(self, G, source, target, k, weight=None):
        """获取两点之间k条最短路径"""
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    def set_vulnerability_lines(self, vuln_lines):
        """设置漏洞相关行，用于关键路径分析时保护这些行"""
        self.vulnerability_lines = set(vuln_lines)
        # 同时更新关键路径分析器的漏洞行信息
        self.critical_path_analyzer.set_vulnerability_lines(vuln_lines)
        
    def get_allpath(self, vulnerability_lines=None):
        global processed_snippets, progress_bar
        global total_nodes_across_dataset, total_critical_nodes_across_dataset
        global total_vuln_nodes_across_dataset, total_centrality_nodes_across_dataset, total_overlap_nodes_across_dataset
        start_time = time.time()
        
        # 设置漏洞相关行
        if vulnerability_lines is not None:
            self.vulnerability_lines = set(vulnerability_lines)
            self.critical_path_analyzer.set_vulnerability_lines(vulnerability_lines)
        
        # 更新进度条 - 只增加计数，实际更新由tqdm的miniters参数控制
        if progress_bar is not None:
            processed_snippets += 1
            
            # 只在每50个代码片段时更新进度条和显示信息
            if processed_snippets % 50 == 0:
                # 更新进度条
                progress_bar.update(50)  # 一次性更新50个
                
                progress_bar.set_postfix({
                    "已处理": processed_snippets, 
                    "总数": total_snippets
                })
        
        # 获取缓存路径
        self._get_cache_paths()
        
        # 尝试从缓存加载完整的路径结果
        if os.path.exists(self.path_cache_path):
            try:
                with open(self.path_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    # 验证缓存数据的完整性
                    if ('num_path' in cached_data and 'mapped_paths' in cached_data and 
                        'dece_node_count' in cached_data):

                        graph_node_count = cached_data.get('graph_node_count')
                        critical_node_count = cached_data.get('critical_node_count')
                        vuln_node_count = cached_data.get('vuln_node_count')
                        centrality_node_count = cached_data.get('centrality_node_count')
                        overlap_node_count = cached_data.get('overlap_node_count')
                        centrality_includes_overlap = cached_data.get('centrality_includes_overlap')

                        if (graph_node_count is None or critical_node_count is None or
                            vuln_node_count is None or centrality_node_count is None or
                            overlap_node_count is None or centrality_includes_overlap is not True):
                            critical_nodes = self.critical_path_analyzer.identify_critical_nodes_by_centrality(
                                self.G, self.firstlineno, self.finlineno, vulnerability_lines, threshold=0.7
                            )
                            graph_node_count = sum(1 for n in self.G.nodes() if n != 0)
                            vuln_nodes_in_graph = set()
                            if vulnerability_lines:
                                vuln_nodes_in_graph = {int(v) for v in vulnerability_lines if v in self.G.nodes() and v != 0}

                            centrality_selected_nodes = set(getattr(self.critical_path_analyzer, 'last_centrality_selected_nodes', set()))
                            if not centrality_selected_nodes:
                                centrality_selected_nodes = set(critical_nodes)
                            centrality_selected_nodes = centrality_selected_nodes - {self.firstlineno} - set(self.finlineno)
                            centrality_selected_nodes = {n for n in centrality_selected_nodes if n != 0}

                            overlap_nodes = centrality_selected_nodes & vuln_nodes_in_graph
                            centrality_nodes = centrality_selected_nodes

                            vuln_node_count = len(vuln_nodes_in_graph)
                            centrality_node_count = len(centrality_nodes)
                            overlap_node_count = len(overlap_nodes)
                            critical_node_count = vuln_node_count + centrality_node_count - overlap_node_count

                            cached_data['graph_node_count'] = graph_node_count
                            cached_data['critical_node_count'] = critical_node_count
                            cached_data['vuln_node_count'] = vuln_node_count
                            cached_data['centrality_node_count'] = centrality_node_count
                            cached_data['overlap_node_count'] = overlap_node_count
                            cached_data['centrality_includes_overlap'] = True
                            try:
                                with open(self.path_cache_path, 'wb') as wf:
                                    pickle.dump(cached_data, wf)
                            except Exception:
                                pass

                        total_nodes_across_dataset += int(graph_node_count)
                        total_critical_nodes_across_dataset += int(critical_node_count)
                        total_vuln_nodes_across_dataset += int(vuln_node_count)
                        total_centrality_nodes_across_dataset += int(centrality_node_count)
                        total_overlap_nodes_across_dataset += int(overlap_node_count or 0)

                        return (cached_data['num_path'], cached_data['mapped_paths'], 
                                cached_data['dece_node_count'])
            except Exception:
                pass
        
        # 如果指定漏洞行，就使用
        if vulnerability_lines:
            self.critical_path_analyzer.set_vulnerability_lines(vulnerability_lines)
        else:
            # 如果未指定漏洞行，则使用空集
            self.critical_path_analyzer.set_vulnerability_lines(set())

        # 记录原始图的节点和边数量
        self.original_graph_nodes = len(self.G.nodes())
        self.original_graph_edges = len(self.G.edges())

        # 使用关键路径分析器生成关键路径
        critical_path, critical_nodes = self.critical_path_analyzer.analyze_and_generate_path(
            self.G, self.firstlineno, self.finlineno, vulnerability_lines
        )

        graph_node_count = sum(1 for n in self.G.nodes() if n != 0)
        vuln_nodes_in_graph = set()
        if vulnerability_lines:
            vuln_nodes_in_graph = {int(v) for v in vulnerability_lines if v in self.G.nodes() and v != 0}

        centrality_selected_nodes = set(getattr(self.critical_path_analyzer, 'last_centrality_selected_nodes', set()))
        if not centrality_selected_nodes:
            centrality_selected_nodes = set(critical_nodes)
        centrality_selected_nodes = centrality_selected_nodes - {self.firstlineno} - set(self.finlineno)
        centrality_selected_nodes = {n for n in centrality_selected_nodes if n != 0}

        overlap_nodes = centrality_selected_nodes & vuln_nodes_in_graph
        centrality_nodes = centrality_selected_nodes

        vuln_node_count = len(vuln_nodes_in_graph)
        centrality_node_count = len(centrality_nodes)
        overlap_node_count = len(overlap_nodes)
        critical_node_count = vuln_node_count + centrality_node_count - overlap_node_count

        total_nodes_across_dataset += int(graph_node_count)
        total_critical_nodes_across_dataset += int(critical_node_count)
        total_vuln_nodes_across_dataset += int(vuln_node_count)
        total_centrality_nodes_across_dataset += int(centrality_node_count)
        total_overlap_nodes_across_dataset += int(overlap_node_count)

        # 获取路径
        all_paths = []

        # 关键节点路径（覆盖最多关键节点的路径），关键路径在critical_path_analysis.py中生成
        mapped_critical_path = []
        for node in critical_path:
            if node in self.func_name:
                mapped_critical_path.append(self.func_name[node])
            else:
                mapped_critical_path.append(node)

        all_paths.append(mapped_critical_path)

        # 缓存结果
        try:
            cache_data = {
                'num_path': len(all_paths),
                'mapped_paths': all_paths,
                'dece_node_count': len(self.dece_node),
                'graph_node_count': graph_node_count,
                'critical_node_count': critical_node_count,
                'vuln_node_count': vuln_node_count,
                'centrality_node_count': centrality_node_count,
                'overlap_node_count': overlap_node_count,
                'centrality_includes_overlap': True
            }
            with open(self.path_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass  # 缓存失败时静默处理

        return len(all_paths), all_paths, len(self.dece_node)

    def run(self, root):
        # self.visit(root)
        self.clean_code = root.text.decode('utf-8')
        self.finlineno.append(root.end_point[0] + 1)
        self.ast_visit(root)

    def parse_ast_file(self, ast_code):
        self.run(ast_code)
        return ast_code

    def parse_ast(self, source_ast):
        self.run(source_ast)
        return source_ast

    def get_source(self, fn):
        ''' 返回给定文件名的文件全部内容。
            几乎完全复制自 stc。 '''
        try:
            f = open(fn, 'r')
            s = f.read()
            f.close()
            return s
        except IOError:
            return ''

    def ast_visit(self, node):
        method = getattr(self, "visit_" + node.type)
        return method(node)

# 遍历抽象语法树（AST）来构建控制流图（CFG）
    def visit_program(self, node):
        # self.finlineno.append(node.end_point[0] + 1)
        self.finlineno.append(node.children[-1].end_point[0] + 1)
        for index, z in enumerate(node.children):
            for i in range(z.start_point[0] + 1, z.end_point[0] + 2):
                self.G.add_node(i)
            if self.firstlineno > z.start_point[0] + 1:
                self.firstlineno = z.start_point[0] + 1
            if z.type == "compound_statement":
                if index == len(node.children) - 1:
                    self.finlineno.append(z.end_point[0] + 1)
                self.ast_visit(z)
            if z.type == "local_variable_declaration":
                self.ast_visit(z)
            else:
                self.visit_piece(z)

    def visit_translation_unit(self, node):
        self.finlineno.append(node.children[-1].end_point[0] + 1)
        # for i in range(node.start_point[0] + 1, node.end_point[0] + 2):
        #     self.G.add_edge(i, i + 1, weight=1)
        for index, z in enumerate(node.children):
            for i in range(z.start_point[0] + 1, z.end_point[0] + 2):
                self.G.add_node(i)
            if self.firstlineno > z.start_point[0] + 1:
                self.firstlineno = z.start_point[0] + 1
            if z.type == "function_definition":
                if index == len(node.children) - 1:
                    self.finlineno.append(z.end_point[0] + 1)
                self.ast_visit(z)
            else:
                self.visit_piece(z)

    def visit_function_definition(self, node):
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                
    def visit_ERROR(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.start_point[0] != node.end_point[0]:
            for j in range(node.start_point[0], node.end_point[0] + 1):
                self.G.add_edge(j, j + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)
            
    def visit_function_declarator(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_expression_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_declaration(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_return_statement(self, node):
        if node.end_point[0] == self.finlineno[0]:
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
            self.G.add_edge(node.start_point[0], node.end_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for i in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(i, i + 1, weight=1)
        if node.end_point[0] + 1 not in self.finlineno:
            self.finlineno.append(node.end_point[0] + 1)
            
    def visit_assert_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] + 1 > node.start_point[0] + 1:
            for i in range(node.start_point[0] + 1, node.end_point[0] + 1):
                self.G.add_edge(i, i + 1, weight=1)
        if node.end_point[0] + 1 not in self.finlineno:
            self.finlineno.append(node.end_point[0] + 1)
            
    def visit_goto_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_object_creation_expression(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_case_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)
            
    def visit_if_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:  # 命名子节点计数
            self.G.add_edge(node.start_point[0]+1, node.next_sibling.start_point[0]+1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        if node.end_point[0] + 1 != self.finlineno[0]:
            self.G.add_edge(node.children[-1].end_point[0] + 1, node.end_point[0] + 2, weight=1)
        for z in node.children:
            if z.type == "else":
                self.dece_node.append(z.start_point[0] + 1)
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
            elif z.type == "compound_statement":
                if node.next_sibling is not None:
                    self.G.add_edge(z.end_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                
    def visit_for_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.circle.append((node.start_point[0] + 1, node.end_point[0] + 1))
        self.dece_node.append(node.start_point[0] + 1)
        # 添加 'For condiation' 语句
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            elif z.type == "if_statement":
                for j in z.children:
                    if j.type == "compound_statement":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == "try_statement":
                for j in z.children:
                    if j.type == "compound_statement":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == "switch_statement":
                for j in z.children:
                    if j.type == "switch_block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
                    elif j.type == "compound_statement":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            else:
                self.visit_piece(z)
                self.G.add_edge(z.end_point[0]+1, node.start_point[0]+1, weight=1)
        self.loopflag = node.end_point[0] + 1
        
    def visit_do_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:  # 命名子节点计数
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        if node.end_point[0] + 1 != self.finlineno[0]:
            self.G.add_edge(node.children[-1].end_point[0] + 1, node.end_point[0] + 2, weight=1)
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                
    def visit_try_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        body_node = {}
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
                body_node['bs'] = z.start_point[0] + 1
                body_node['be'] = z.end_point[0] + 1
            elif z.type == "finally_clause":
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
                self.dece_node.append(z.start_point[0] + 1)
                body_node['fs'] = z.start_point[0] + 1
                body_node['fe'] = z.end_point[0] + 1
                self.ast_visit(z)
            elif z.type == "catch_clause":
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
                self.dece_node.append(z.start_point[0] + 1)
                body_node['cs'] = z.start_point[0] + 1
                body_node['ce'] = z.end_point[0] + 1
                self.ast_visit(z)
            else:
                self.visit_piece(z)
        if 'bs' in body_node and 'cs' in body_node:
            self.G.add_edge(body_node['be'], body_node['cs'], weight=1)
        if 'cs' in body_node and 'fs' in body_node:
            self.G.add_edge(body_node['ce'], body_node['fs'], weight=1)
            
    def visit_while_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.circle.append((node.start_point[0] + 1, node.end_point[0] + 1))
        self.dece_node.append(node.start_point[0] + 1)
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                self.G.add_edge(z.end_point[0]+1, node.start_point[0]+1, weight=1)
        self.loopflag = node.end_point[0] + 1
        
    def visit_switch_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        for z in node.children:
            if z.type == "switch_block":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                
    def visit_throw_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
        if node.end_point[0] + 1 not in self.finlineno:
            self.finlineno.append(node.end_point[0] + 1)
            
    def visit_break_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
        if self.loopflag != 0:
            self.G.add_edge(node.start_point[0], self.loopflag, weight=1)
            
    def visit_continue_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
        for i in self.circle:
            if i[0] <= node.start_point[0] <= i[1]:
                self.G.add_edge(node.start_point[0], i[0], weight=1)
                
    def visit_parenthesized_expression(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_local_variable_declaration(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_switch_block(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            if z.type == "switch_block_statement_group":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                
    def visit_switch_block_statement_group(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)
            
    def visit_catch_clause(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                
    def visit_finally_clause(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                
    def visit_pointer_declarator(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_preproc_if(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_preproc_ifdef(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_preproc_elif(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_preproc_else(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_preproc_def(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_preproc_function_def(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_preproc_params(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_preproc_call(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_class_declaration(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    def visit_enhanced_for_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.circle.append((node.start_point[0] + 1, node.end_point[0] + 1))
        self.dece_node.append(node.start_point[0] + 1)
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                self.G.add_edge(z.end_point[0]+1, node.start_point[0]+1, weight=1)
        self.loopflag = node.end_point[0] + 1
        
    def visit_labeled_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)
            
    def visit_synchronized_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for index, z in enumerate(node.children):
            if z.type == "compound_statement":
                self.ast_visit(z)
                
    def visit_try_with_resources_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            else:
                self.visit_piece(z)
                
    # 添加一些通用的访问方法，处理其他可能的节点类型
    def visit_generic(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)
                
    # 添加一些空方法，处理不需要特殊处理的节点类型
    def visit_identifier(self, node):
        pass
        
    def visit_generic_type(self, node):
        pass
        
    def visit_for(self, node):
        pass

    def visit_compound_statement(self, node):
        # self.G.add_edge(node.start_point[0]+1, node.end_point[0]+1)
        for index, z in enumerate(node.children):
            if z.type == "for_statement":
                self.ast_visit(z)
            elif z.type == "enhanced_for_statement":
                self.ast_visit(z)
            elif z.type == "while_statement":
                self.ast_visit(z)
            elif z.type == "do_statement":
                self.ast_visit(z)
            elif z.type == "try_with_resources_statement":
                self.ast_visit(z)
            elif z.type == "assert_statement":
                self.ast_visit(z)
            elif z.type == "switch_statement":
                self.ast_visit(z)
            elif z.type == "case_statement":
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
                self.ast_visit(z)
            elif z.type == "switch_block":
                self.ast_visit(z)
            elif z.type == "switch_block_statement_group":
                self.ast_visit(z)
            elif z.type == "labeled_statement":
                self.ast_visit(z)
            elif z.type == "continue_statement":
                self.ast_visit(z)
            elif z.type == "try_statement":
                self.ast_visit(z)
            elif z.type == "throw_statement":
                self.ast_visit(z)
            elif z.type == "if_statement":
                self.ast_visit(z)
            elif z.type == "synchronized_statement":
                self.ast_visit(z)
            elif z.type == "expression_statement":
                self.ast_visit(z)
            elif z.type == "local_variable_declaration":
                self.ast_visit(z)
            elif z.type == "return_statement":
                self.ast_visit(z)
            elif z.type == "compound_statement":
                self.ast_visit(z)
            elif z.type == "parenthesized_expression":
                self.ast_visit(z)
            elif z.type == "ERROR":
                self.visit_ERROR(z)
            elif z.type == "break_statement":
                self.ast_visit(z)
            elif z.type == "class_declaration":
                self.ast_visit(z)
            elif z.type == "declaration":
                self.ast_visit(z)
            elif z.type == "function_declarator":
                self.ast_visit(z)
            elif z.type == "}":
                self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
            elif z.type == "{":
                self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
            elif z.type == ";":
                self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
            else:
                self.visit_piece(z)
        if len(node.children) > 0 and node.children[0].type == "{":
            self.G.add_edge(node.children[0].start_point[0] + 1, node.children[0].start_point[0] + 2, weight=1)
        if len(node.children) > 0 and node.children[-1].type == "}":
            self.G.add_edge(node.children[-1].start_point[0], node.children[-1].start_point[0] + 1, weight=1)

    def visit_piece(self, node):
        # self.G.add_edge(node.start_point[0]+1, node.end_point[0]+1)
        if node.type == "for_statement":
            self.ast_visit(node)
        elif node.type == "enhanced_for_statement":
            self.ast_visit(node)
        elif node.type == "while_statement":
            self.ast_visit(node)
        elif node.type == "do_statement":
            self.ast_visit(node)
        elif node.type == "try_with_resources_statement":
            self.ast_visit(node)
        elif node.type == "assert_statement":
            self.ast_visit(node)
        elif node.type == "switch_statement":
            self.ast_visit(node)
        elif node.type == "switch_block":
            self.ast_visit(node)
        elif node.type == "switch_block_statement_group":
            self.ast_visit(node)
        elif node.type == "labeled_statement":
            self.ast_visit(node)
        elif node.type == "continue_statement":
            self.ast_visit(node)
        elif node.type == "try_statement":
            self.ast_visit(node)
        elif node.type == "throw_statement":
            self.ast_visit(node)
        elif node.type == "if_statement":
            self.ast_visit(node)
        elif node.type == "synchronized_statement":
            self.ast_visit(node)
        elif node.type == "expression_statement":
            self.ast_visit(node)
        elif node.type == "local_variable_declaration":
            self.ast_visit(node)
        elif node.type == "parenthesized_expression":
            self.ast_visit(node)
        elif node.type == "return_statement":
            self.ast_visit(node)
        elif node.type == "ERROR":
            self.visit_ERROR(node)
        elif node.type == "break_statement":
            self.ast_visit(node)
        elif node.type == "class_declaration":
            self.ast_visit(node)
        elif node.type == "declaration":
            self.ast_visit(node)
        elif node.type == "function_declarator":
            self.ast_visit(node)
        elif node.type == "compound_statement":
            self.ast_visit(node)
        elif node.type == "goto_statement":
            self.ast_visit(node)
        elif node.type == "preproc_if":
            self.ast_visit(node)
        elif node.type == "preproc_params":
            self.ast_visit(node)
        elif node.type == "pointer_declarator":
            self.ast_visit(node)
        elif node.type == "preproc_ifdef":
            self.ast_visit(node)
        elif node.type == "preproc_elif":
            self.ast_visit(node)
