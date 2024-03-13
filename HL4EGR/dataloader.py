import torch
from datautil import load_rating_file_to_list, build_gg_sim_gh, build_ui_hyper_graph, build_gu_hyper_graph, \
    load_rating_file_to_matrix, load_negative_file, load_group_member_to_dict
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from torch.utils import data


class GroupDataset(data.Dataset):
    # 返回训练群组用的数据集，和用户训练集、群组测试集
    def __init__(self, dataset="Mafengwo"):
        print(f"[{dataset.upper()}] loading...")

        user_path = f"../data/{dataset}/userRating"
        group_path = f"../data/{dataset}/groupRating"
        self.user_total_matrix = load_rating_file_to_matrix(user_path + ".txt")
        self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt")
        self.user_test_ratings = load_rating_file_to_list(user_path + "Test.txt")
        self.num_users, self.num_items = self.user_total_matrix.shape

        print(f"Total UserItem {self.user_total_matrix.shape} with {len(self.user_total_matrix.keys())} interactions, "
              f"sparsity: {(1 - (len(self.user_total_matrix.keys()) / self.num_users / self.num_items)):.5f}")

        self.group_train_matrix = load_rating_file_to_matrix(group_path + "Train.txt")
        self.group_val_ratings = load_rating_file_to_list(group_path + "Val.txt")
        self.group_test_ratings = load_rating_file_to_list(group_path + "Test.txt")
        self.group_member_dict = load_group_member_to_dict(f"../data/{dataset}/groupMember.txt")
        self.num_groups = self.group_member_dict.__len__()
        self.num_train_group, _ = self.group_train_matrix.shape
        self.num_test_group = self.num_groups - self.num_train_group

        self.group2items = defaultdict(list)
        for (g, i) in self.group_train_matrix.keys():
            self.group2items[g].append(i)
        self.users_feat = self.user_total_dataloader(batch_size=128, test_mode=True)  # 不对
        # u-i超图
        self.user_item_hyper_graph = build_ui_hyper_graph(user_path + "Train.txt", self.num_users, self.num_items)
        self.group_user_hyper_graph = build_gu_hyper_graph(f"../data/{dataset}/groupMemberTrain.txt", self.num_groups,
                                                           self.num_users)
        self.group_group_sim_item = build_gg_sim_gh(f"../data/{dataset}/group_sim_item_train.txt",
                                                    self.num_groups)
        self.group_group_sim_prefer = build_gg_sim_gh(f"../data/{dataset}/group_sim_prefer_train.txt",
                                                      self.num_groups)

    def __len__(self):
        return self.num_train_group

    def __getitem__(self, index):
        group_feat = np.zeros((1, self.num_items))
        num_corrupt = 5
        max_len = max([len(members) for members in self.group_member_dict.values()])
        group_id = index
        members = self.group_member_dict[group_id]
        group_mask = torch.zeros((max_len,))
        group_mask[len(members):] = -np.inf
        if len(members) < max_len:
            group_user = torch.hstack((torch.LongTensor(members), torch.zeros((max_len - len(members),)))).long()
        else:
            group_user = torch.LongTensor(members)

        corrupted_user = []
        for j in range(num_corrupt):
            random_u = np.random.randint(self.num_users)
            while random_u in members:
                random_u = np.random.randint(self.num_users)
            corrupted_user.append(random_u)
        corrupted_user = np.array(corrupted_user)
        user_item = self.users_feat[group_user]
        couser_item = self.users_feat[corrupted_user]
        group_feat[0, self.group2items[group_id]] = 1.0
        gruop_item = group_feat[0]
        return group_id, group_user, corrupted_user, group_mask, user_item, couser_item, gruop_item

    def user_pretrain_dataloader(self, batch_size, test_mode=False):
        user2item = defaultdict(list)
        for (u, i) in self.user_train_matrix.keys():
            user2item[u].append(i)

        user_feat = np.zeros((self.num_users, self.num_items))
        users = []
        for user_id in range(self.num_users):
            items = user2item[user_id]
            user_feat[user_id, items] = 1.0
            users.append(int(user_id))
        if test_mode:
            return torch.FloatTensor(user_feat)
        train_data = TensorDataset(torch.LongTensor(users), torch.FloatTensor(user_feat))
        return DataLoader(train_data, batch_size=batch_size, shuffle=False)

    def user_total_dataloader(self, batch_size, test_mode=False):
        user2item = defaultdict(list)
        for (u, i) in self.user_total_matrix.keys():
            user2item[u].append(i)

        user_feat = np.zeros((self.num_users, self.num_items))
        for user_id in range(self.num_users):
            items = user2item[user_id]
            user_feat[user_id, items] = 1.0
        if test_mode:
            return torch.FloatTensor(user_feat)
        train_data = TensorDataset(torch.FloatTensor(user_feat))
        return DataLoader(train_data, batch_size=batch_size, shuffle=False)

    def test_group_dataloader(self, batch_size):
        group2items = defaultdict(list)
        for (g, i) in self.group_train_matrix.keys():
            group2items[g].append(i)

        max_len = max([len(members) for members in self.group_member_dict.values()])
        users_feat = self.users_feat  # User features

        all_group_users, all_group_mask, all_user_items, all_group_user, all_group_id = [], [], [], [], []
        print("finish group_dict")
        test_groups = [rating[0] for rating in self.group_test_ratings]
        for group_id in test_groups:
            members = self.group_member_dict[group_id]

            # Mask for labeling valid group members
            group_mask = torch.zeros((max_len,))
            group_mask[len(members):] = -np.inf
            if len(members) < max_len:
                group_user = torch.hstack((torch.LongTensor(members), torch.zeros((max_len - len(members),)))).long()
            else:
                group_user = torch.LongTensor(members)
            all_group_id.append(group_id)
            all_group_mask.append(group_mask)
            all_user_items.append(users_feat[group_user])
            all_group_user.append(group_user)

        train_data = TensorDataset(torch.LongTensor(all_group_id),
                                   torch.stack(all_group_user),
                                   torch.stack(all_group_mask))
        # print("begin tensor")
        return DataLoader(train_data, batch_size=batch_size, shuffle=False)

    def group_member_msg(self):
        all_group_mask, all_group_users = [], []
        max_len = max([len(members) for members in self.group_member_dict.values()])
        for group_id in range(self.num_groups):
            members = self.group_member_dict[group_id]

            # Mask for labeling valid group members
            group_mask = torch.zeros((max_len,))
            group_mask[len(members):] = -np.inf
            if len(members) < max_len:
                group_user = torch.hstack((torch.LongTensor(members), torch.zeros((max_len - len(members),)))).long()
            else:
                group_user = torch.LongTensor(members)
            all_group_users.append(group_user)
            all_group_mask.append(group_mask)
        return all_group_users, all_group_mask






