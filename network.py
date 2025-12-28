# -*- coding: utf-8 -*-
"""NETWORK
Modified for CI/CD Assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import nltk

# 确保 NLTK 数据下载
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn

# ==========================================
# 第一部分: PageRank Network
# ==========================================
print("Generating PageRank Graph...")

# 有向グラフ作成（10ノード）
G = nx.DiGraph()
G.add_edges_from([
    # メインサイクル
    (0, 1), (1, 2), (2, 0),
    # サブサイクル
    (2, 3), (3, 4), (4, 2),
    # ハブ構造（5が複数へリンク）
    (5, 1), (5, 2), (5, 6), (5, 7),
    # 双方向リンク
    (6, 7), (7, 6),
    # 周辺ノード（スプラウト構造）
    (8, 5), (8, 9),
    (9, 8), (9, 3),
])

# PageRank計算
pr = nx.pagerank(G, alpha=0.85)

# PageRank値をノードサイズに反映（スケーリング）
node_size = [v * 5000 for v in pr.values()]

# レイアウト設定（spring_layoutで自然な配置）
pos = nx.spring_layout(G, seed=42, k=0.6)

# グラフ描画
plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="lightsteelblue", alpha=0.9)
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1.2)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

# PageRank値をノードの下に表示
for node, (x, y) in pos.items():
    plt.text(x, y - 0.08, f"{pr[node]:.3f}", fontsize=9, ha='center', color='darkred')

plt.title("Complex Directed Network with PageRank", fontsize=14)
plt.axis("off")

# 【修改点】：保存图片而不是显示
plt.savefig("pagerank_graph.png")
print("Saved pagerank_graph.png")
plt.close() # 关闭画布，防止重叠


# ==========================================
# 第二部分: WordNet Graph
# ==========================================
print("Generating WordNet Graph...")

def build_wordnet_graph(word: str, pos=None, max_hyponyms=5):
    """
    构建 WordNet 关系图
    """
    G_wn = nx.DiGraph()

    # 中心ノード：生の単語
    center_node = f'"{word}"'
    G_wn.add_node(center_node, type='word', label=word)

    # synsets を取得
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        print(f'No synsets found for "{word}"')
        return G_wn

    for s in synsets:
        syn_node = s.name()
        gloss = s.definition()

        # synset ノード
        G_wn.add_node(syn_node, type='synset', label=f'{syn_node}\n({gloss})')
        G_wn.add_edge(center_node, syn_node, relation='sense')

        # 上位語（hypernyms）
        for h in s.hypernyms():
            h_node = h.name()
            G_wn.add_node(h_node, type='hypernym', label=f'{h_node}\n({h.definition()})')
            G_wn.add_edge(syn_node, h_node, relation='hypernym')

        # 下位語（hyponyms）
        hypos = s.hyponyms()[:max_hyponyms]
        for hy in hypos:
            hy_node = hy.name()
            G_wn.add_node(hy_node, type='hyponym', label=f'{hy_node}\n({hy.definition()})')
            G_wn.add_edge(syn_node, hy_node, relation='hyponym')

    return G_wn

def draw_wordnet_graph(G_wn, figsize=(12, 8), filename="wordnet_graph.png"):
    """
    绘制并保存 WordNet 图
    """
    if len(G_wn.nodes) == 0:
        print("Graph is empty.")
        return

    plt.figure(figsize=figsize)

    pos = nx.spring_layout(G_wn, k=0.8, iterations=50)

    # 节点分类
    word_nodes   = [n for n, d in G_wn.nodes(data=True) if d.get('type') == 'word']
    synset_nodes = [n for n, d in G_wn.nodes(data=True) if d.get('type') == 'synset']
    hyper_nodes  = [n for n, d in G_wn.nodes(data=True) if d.get('type') == 'hypernym']
    hypo_nodes   = [n for n, d in G_wn.nodes(data=True) if d.get('type') == 'hyponym']

    nx.draw_networkx_nodes(G_wn, pos, nodelist=word_nodes,   node_size=800, node_color='lightcoral', label='word')
    nx.draw_networkx_nodes(G_wn, pos, nodelist=synset_nodes, node_size=700, node_color='lightblue',  label='synset')
    nx.draw_networkx_nodes(G_wn, pos, nodelist=hyper_nodes,  node_size=600, node_color='lightgreen', label='hypernym')
    nx.draw_networkx_nodes(G_wn, pos, nodelist=hypo_nodes,   node_size=600, node_color='khaki',      label='hyponym')

    nx.draw_networkx_edges(G_wn, pos, arrows=True, arrowstyle='-|>', arrowsize=12)

    # 标签处理
    labels = {}
    for n, d in G_wn.nodes(data=True):
        full_label = d.get('label', str(n))
        first_line = full_label.split('\n')[0]
        labels[n] = first_line

    nx.draw_networkx_labels(G_wn, pos, labels=labels, font_size=8)

    plt.axis('off')
    plt.legend(scatterpoints=1)
    plt.tight_layout()
    
    # 【修改点】：保存图片
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

# 运行 WordNet 部分
query_word = "man"
G_wordnet = build_wordnet_graph(query_word, pos=None, max_hyponyms=5)
draw_wordnet_graph(G_wordnet)