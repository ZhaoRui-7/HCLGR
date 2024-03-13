import numpy as np
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch


def load_rating_file_to_list(filename):
    ratings = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        content = line.split()
        ratings.append([int(content[0]), int(content[1])])
    return ratings


def load_rating_file_to_matrix(filename):
    num_users, num_items = 0, 0

    lines = open(filename, 'r').readlines()

    for line in lines:
        content = line.split()
        u, i = int(content[0]), int(content[1])
        num_users = max(num_users, u)
        num_items = max(num_items, i)

    mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)

    for line in lines:
        content = line.split()
        if len(content) > 2:
            u, i, rating = int(content[0]), int(content[1]), int(content[2])
            if rating > 0:
                mat[u, i] = 1.0
        else:
            u, i = int(content[0]), int(content[1])
            mat[u, i] = 1.0
    return mat


def load_negative_file(filename):
    negatives = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        neg = line.split()[1:]
        neg = [int(item) for item in neg]
        negatives.append(neg)
    return negatives


def load_group_member_to_dict(user_in_group_path):
    group_member_dict = defaultdict(list)

    lines = open(user_in_group_path, 'r').readlines()

    for line in lines:
        content = line.split()
        group = int(content[0])
        for member in content[1].split(','):
            group_member_dict[group].append(int(member))

    return group_member_dict

def build_ui_hyper_graph(path,num_edge,num_node):
    """Return user-item hyper-graph"""
    user_item_dict = defaultdict(list)

    for line in open(path, 'r').readlines():
        contents = line.split()
        user, item = int(contents[0]), int(contents[1])
        user_item_dict[user].append(item)

    def _prepare(user_dict, rows, axis=0):
        nodes, edges = [], []

        for user_id in range(num_edge):
            edges.extend([user_id] * len(user_dict[user_id]))
            nodes.extend(user_dict[user_id])

        hyper_graph = csr_matrix((np.ones(len(nodes)), (nodes, edges)), shape=(rows, num_edge))
        hyper_deg = np.array(hyper_graph.sum(axis=axis)).squeeze()
        hyper_deg[hyper_deg == 0.] = 1
        hyper_deg = sp.diags(1.0 / hyper_deg)
        return hyper_graph, hyper_deg

    item_hg, item_hg_deg = _prepare(user_item_dict, num_node)


    item_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(item_hg_deg),
                                       convert_sp_mat_to_sp_tensor(item_hg).t())

    print(
        f"User-Item hyper-graph {item_hyper_graph.shape}")

    return item_hyper_graph

def build_gu_hyper_graph(path,num_edge,num_node):
    """Return user-item hyper-graph"""
    group_member_dict = defaultdict(list)

    for line in open(path, 'r').readlines():
        content = line.split()
        group = int(content[0])
        for member in content[1].split(','):
            group_member_dict[group].append(int(member))

    def _prepare(user_dict, rows, axis=0):
        nodes, edges = [], []

        for user_id in range(num_edge):
            edges.extend([user_id] * len(user_dict[user_id]))
            nodes.extend(user_dict[user_id])

        hyper_graph = csr_matrix((np.ones(len(nodes)), (nodes, edges)), shape=(rows, num_edge))
        hyper_deg = np.array(hyper_graph.sum(axis=axis)).squeeze()
        hyper_deg[hyper_deg == 0.] = 1
        hyper_deg = sp.diags(1.0 / hyper_deg)
        return hyper_graph, hyper_deg

    item_hg, item_hg_deg = _prepare(group_member_dict, num_node)


    item_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(item_hg_deg),
                                       convert_sp_mat_to_sp_tensor(item_hg).t())

    print(
        f"Group-User hyper-graph {item_hyper_graph.shape}")

    return item_hyper_graph

def build_gg_sim_gh(path, num_groups):
    group_member_dict = defaultdict(list)

    for line in open(path, 'r').readlines():
        contents = line.split()
        group_id, sim_group_id = int(contents[0]), int(contents[1])
        group_member_dict[group_id].append(sim_group_id)

    def _prepare(group_member_dict, rows, axis=0):
        nodes, edges = [], []

        for group_id in range(num_groups):
            edges.extend([group_id] * len(group_member_dict[group_id]))
            nodes.extend(group_member_dict[group_id])

        hyper_graph = csr_matrix((np.ones(len(nodes)), (nodes, edges)), shape=(rows, num_groups))
        hyper_deg = np.array(hyper_graph.sum(axis=axis)).squeeze()
        hyper_deg[hyper_deg == 0.] = 1
        hyper_deg = sp.diags(1.0 / hyper_deg)
        return hyper_graph, hyper_deg

    item_hg, item_hg_deg = _prepare(group_member_dict, num_groups)

    item_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(item_hg_deg),
                                       convert_sp_mat_to_sp_tensor(item_hg).t())

    print(
        f"Group-User hyper-graph {item_hyper_graph.shape}")

    return item_hyper_graph

def convert_sp_mat_to_sp_tensor(x):
    """Convert `csr_matrix` into `torch.SparseTensor` format"""
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
