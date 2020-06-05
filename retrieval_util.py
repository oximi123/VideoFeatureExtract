import numpy as np
import glob
import pandas as pd
import pickle
import time

import cv2
import imagehash
import numpy as np
import networkx as nx
from tqdm.notebook import tqdm
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from networkx.algorithms.dag import dag_longest_path

def compute_similarities(query_features, refer_features):
    """
      用于计算两组特征(已经做过l2-norm)之间的相似度
      Args:
        query_features: shape: [N, D]
        refer_features: shape: [M, D]
      Returns:
        sorted_sims: shape: [N, M]
        unsorted_sims: shape: [N, M]
    """
    sorted_sims = []
    unsorted_sims = []
    # 计算待查询视频和所有视频的距离
    dist = np.nan_to_num(cdist(query_features, refer_features, metric='cosine'))
    for i, v in enumerate(query_features):
        # 归一化，将距离转化成相似度
        # sim = np.round(1 - dist[i] / dist[i].max(), decimals=6)
        sim = 1 - dist[i]
        # 按照相似度的从大到小排列，输出index
        unsorted_sims += [sim]
        sorted_sims += [[(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]]
    return sorted_sims, unsorted_sims

def compute_dists(query_features, refer_features):
    """
      用于计算两组特征(已经做过l2-norm)之间的余弦距离
      Args:
        query_features: shape: [N, D]
        refer_features: shape: [M, D]
      Returns:
        idxs: shape [N, M]
        unsorted_dists: shape: [N, M]
        sorted_dists: shape: [N, M]
    """
    sims = np.dot(query_features, refer_features.T)
    unsorted_dists = 1 - sims # sort 不好改降序
    # unsorted_dist = np.nan_to_num(cdist(query_features, refer_features, metric='cosine'))
    idxs = np.argsort(unsorted_dists)
    rows = np.dot(np.arange(idxs.shape[0]).reshape((idxs.shape[0], 1)), np.ones((1, idxs.shape[1]))).astype(int)
    sorted_dists = unsorted_dists[rows, idxs]
    # sorted_dists = np.sort(unsorted_dists)
    return idxs, unsorted_dists, sorted_dists


def get_frame_alignment(query_features, refer_features, top_K=5, min_sim=0.80, max_step=10):
    """
      用于计算两组特征(已经做过l2-norm)之间的帧匹配结果
      Args:
        query_features: shape: [N, D]
        refer_features: shape: [M, D]
        top_K: 取前K个refer_frame
        min_sim: 要求query_frame与refer_frame的最小相似度
        max_step: 有边相连的结点间的最大步长
      Returns:
        path_query: shape: [1, L]
        path_refer: shape: [1, L]
    """
    node_pair2id = {}
    node_id2pair = {}
    node_id2pair[0] = (-1, -1)  # source
    node_pair2id[(-1, -1)] = 0
    node_num = 1

    DG = nx.DiGraph()
    DG.add_node(0)

    idxs, unsorted_dists, sorted_dists = compute_dists(query_features, refer_features)

    # add nodes
    for qf_idx in range(query_features.shape[0]):
        for k in range(top_K):
            rf_idx = idxs[qf_idx][k]
            sim = 1 - sorted_dists[qf_idx][k]
            if sim < min_sim:
                break
            node_id2pair[node_num] = (qf_idx, rf_idx)
            node_pair2id[(qf_idx, rf_idx)] = node_num
            DG.add_node(node_num)
            node_num += 1

    node_id2pair[node_num] = (query_features.shape[0], refer_features.shape[0])  # sink
    node_pair2id[(query_features.shape[0], refer_features.shape[0])] = node_num
    DG.add_node(node_num)
    node_num += 1

    # link nodes

    for i in range(0, node_num - 1):
        for j in range(i + 1, node_num - 1):

            pair_i = node_id2pair[i]
            pair_j = node_id2pair[j]

            if (pair_j[0] > pair_i[0] and pair_j[1] > pair_i[1] and
                    pair_j[0] - pair_i[0] <= max_step and pair_j[1] - pair_i[1] <= max_step):
                qf_idx = pair_j[0]
                rf_idx = pair_j[1]
                DG.add_edge(i, j, weight=1 - unsorted_dists[qf_idx][rf_idx])

    for i in range(0, node_num - 1):
        j = node_num - 1

        pair_i = node_id2pair[i]
        pair_j = node_id2pair[j]

        if (pair_j[0] > pair_i[0] and pair_j[1] > pair_i[1] and
                pair_j[0] - pair_i[0] <= max_step and pair_j[1] - pair_i[1] <= max_step):
            qf_idx = pair_j[0]
            rf_idx = pair_j[1]
            DG.add_edge(i, j, weight=0)

    longest_path = dag_longest_path(DG)
    if 0 in longest_path:
        longest_path.remove(0)  # remove source node
    if node_num - 1 in longest_path:
        longest_path.remove(node_num - 1)  # remove sink node
    path_query = [node_id2pair[node_id][0] for node_id in longest_path]
    path_refer = [node_id2pair[node_id][1] for node_id in longest_path]

    score = 0.0
    for (qf_idx, rf_idx) in zip(path_query, path_refer):
        score += 1 - unsorted_dists[qf_idx][rf_idx]

    return path_query, path_refer, score

def compute_precision_recall(y_true, y_pred, pr=False):
    """
      用于计算测试结果的P-R值
      Args:
        y_true: dict shape: [N, 5]
        y_pred: dict shape: [M, 5]
        pr: need precision and recall
      Returns:
        f1_score
        precision
        recall
    """
    tp = fp = fn = 0
    threshold = 3000

#    for q_vid in y_true:
    for q_vid in y_pred:
        q_ans = y_pred[q_vid]
        q_label = y_true[q_vid]

        if(len(q_ans) == 5):
            if(q_ans[0] == q_label[0] and abs(q_ans[1] - q_label[1]) <= threshold and abs(q_ans[2] - q_label[2]) <= threshold
            and abs(q_ans[3] - q_label[3]) <= threshold and abs(q_ans[4] - q_label[4]) <= threshold):
                tp += 1
            else:
                fp += 1
        else:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    if(pr):
        return f1_score, precision, recall
    else:
        return f1_score