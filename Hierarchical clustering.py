import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Cluster:
    def __init__(self, path):
        self.G = nx.read_gml(path, label='id')
        self.A = nx.to_numpy_array(self.G)  # G的邻接矩阵
        self.num_node = len(self.A)  # 图G的总结点数
        self.num_edge = sum(sum(self.A))  # 2倍边数
        self.c = {}  # 记录所有Q值对应的社团分布

    # 合并社团
    def merge_cluster(self, iter_num,deltaQ, edges, b):
        Smler = []  # 存放合并的两个社团中编号较小的社团编号
        Lager = []  # 存放合并的两个社团中编号较大的社团编号
        max = 0
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                 neighbor_i = set()  # 存放社团邻接点
                 neighbor_j = set()
                 for x in b[i]:
                     neighbor_i.update([n for n in self.G.neighbors(x + 1)])
                 for y in b[j]:
                     neighbor_j.update([n for n in self.G.neighbors(y + 1)])
                 common = len(neighbor_i & neighbor_j)
                 union = len(neighbor_i | neighbor_j)
                 Jaccard = common * 1.0 / union
                 if Jaccard >= max :
                    max = Jaccard
        (Smler, Lager) = np.where(deltaQ == np.max(deltaQ))  # 找出此时detaQ最大的两个社团 并合并

        # 将Smler中社团和Lager中社团两两合并
        for m in range(len(Smler)):
            # 更新edges矩阵
            edges[Lager[m], :] = edges[Smler[m], :] + edges[Lager[m], :]
            edges[Smler[m], :] = 0
            edges[:, Lager[m]] = edges[:, Smler[m]] + edges[:, Lager[m]]
            edges[:, Smler[m]] = 0
            # 合并社团
            b[Lager[m]] = b[Lager[m]] + b[Smler[m]]

        # 删除edges中Smler相关行列
        edges = np.delete(edges, Smler, axis=0)
        edges = np.delete(edges, Smler, axis=1)
        Smler = sorted(list(set(Smler)), reverse=True)  # 逆序排列
        for i in Smler:
            b.remove(b[i])
        # 记录Q与对应社团分布
        self.c[iter_num] = b.copy()
        return edges, b

    # 求Q最大时的社团分布
    def get_best(self, Q):
        max = np.argmax(Q)
        comms = self.c[max]
        return Q[max], comms

    def clustering(self):
        edges = self.A / self.num_edge  # 社团之间边的总数除以图G的总边数
        a = np.sum(edges, axis=0)  # 与社团节点相连的边占总边数的比例
        b = [[i] for i in range(self.num_node)]  # 每轮合并后的社团分布
        Q = []
        iter_num = 0  # 合并次数
        while len(edges) > 1:  # 直到全部社团合并为一个社团
            num_community = len(edges)  # 社团总数
            deltaQ = -np.power(10, 9) * np.ones((self.num_node, self.num_node))
            # 计算deltaQ
            for i in range(num_community - 1):
                for j in range(i + 1, num_community):
                    if edges[i, j] != 0:
                        deltaQ[i, j] = edges[i, j] - a[i] * a[j]

            if np.sum(deltaQ + np.power(10, 9)) == 0:
                break
            # 合并社团
            edges, b = self.merge_cluster(iter_num, deltaQ, edges, b)

            a = np.sum(edges, axis=0)
            # 计算Q值
            q = 0.0
            for n in range(len(edges)):
                q += edges[n, n] - a[n] * a[n]
            Q.append(q)
            iter_num += 1
        # 找出Q最大时对应的社团分布
        max_Q, best_community = self.get_best(Q)
        return max_Q, best_community




# 为同一社团的节点分配相同的索引
def allocate_index(G, community):
    num_node = nx.number_of_nodes(G)  # 图G中总节点数
    classes = [[] for i in range(num_node)]
    for index, com in enumerate(community):
        for q in com:
            classes[q] = index
    return classes


# 绘出社团分布图
def draw_community(G, com):
    classes = allocate_index(G, com)
    pos = nx.spring_layout(G)
    nodes = [i for i in range(1, 35)]
    nlabels = dict(zip(nodes, nodes))
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=classes)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos, nlabels, font_color='yellow')
    plt.show()

Q, community = Cluster('karate.gml').clustering()
G = nx.read_gml('karate.gml', label='id')
print('maxQ为', Q)
print('社团数为', len(community))
# 打印每个社团中的节点
for i in range(len(community)):
    print(community[i])
draw_community(G, community)
