import math
import numpy as np
import model
import dataloader as dataloader
import torch


def get_hit_k(pred_rank, k, test_ratings):
    hit = 0
    for user in range(pred_rank.shape[0]):
        tmp_hit = np.count_nonzero(pred_rank[user] == test_ratings[user][1])
        hit += tmp_hit
    hit = hit / pred_rank.shape[0]
    return round(hit, 5)
def group_get_hit_k(pred_rank, k, test_ratings, start, epoch):
    hit = 0
    for user in range(pred_rank.shape[0]):
        tmp_hit = np.count_nonzero(pred_rank[user] == test_ratings[start+user][1])
        hit += tmp_hit
    return round(hit, 5)

def get_ndcg_k(pred_rank, k, test_ratings):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == test_ratings[user][1]:
                ndcgs[user] = math.log(2) / math.log(j+2)
    return np.round(np.mean(ndcgs), decimals=5)
def group_get_ndcg_k(pred_rank, k, test_ratings, start):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == test_ratings[start+user][1]:
                ndcgs[user] = math.log(2) / math.log(j+2)
    return np.round(np.mean(ndcgs), decimals=5)

def user_model_leave_one_test(rec_model: model.HCLGR, dataset: dataloader.GroupDataset, test_ratings, device, k_list=None):
    rec_model.eval()
    user_hits, user_ndcgs = [], []


    test_users = [rating[0] for rating in test_ratings]
    user_feat = dataset.user_pretrain_dataloader(batch_size=256, test_mode=True).to(device)

    test_logits,_ = rec_model.user_item_hgcn.user_forward(test_users)
    for k in k_list:
        _,topk_test_group_logits = torch.topk(test_logits,k,dim=1, largest=True)
        topk_test_group_logits = topk_test_group_logits.detach().cpu().numpy()
        user_hits.append(get_hit_k(topk_test_group_logits, k, test_ratings))
        user_ndcgs.append(get_ndcg_k(topk_test_group_logits, k, test_ratings))

    return user_hits, user_ndcgs


def group_model_leave_one_test(rec_model: model.HCLGR, dataset: dataloader.GroupDataset, test_ratings, all_user_emb, all_group_embeds, epoch, device, k_list=None):
    rec_model.eval()

    hits, ndcgs = [], []
    hits_k = [[] for _ in range(5)]
    ndcgs_k = [[] for _ in range(5)]

    bsz = 128
    batch_num = 0
    gi_data = dataset.test_group_dataloader(bsz)
    for batch_data in gi_data:
        batch_data = [x.to(device) for x in batch_data]
        (group_id, group_users,group_masks) = batch_data
        test_group_logits, _, _,_,_,_,_,_ = rec_model(all_user_emb,all_group_embeds,group_id,group_users,group_masks)
        start = bsz*batch_num
        for k, i in zip(k_list, range(len(k_list))):
            _, my_pred_rank = torch.topk(test_group_logits, k, dim=1, largest=True)
            my_pred_rank = my_pred_rank.data.cpu().numpy()
            hits_k[i].append(group_get_hit_k(my_pred_rank, k, test_ratings, start, epoch))
            ndcgs_k[i].append(len(group_id) * group_get_ndcg_k(my_pred_rank, k, test_ratings, start))
        batch_num+=1

    for k, i in zip(k_list, range(len(k_list))):
        hits.append(sum(hits_k[i]) / len(test_ratings))
        ndcgs.append(sum(ndcgs_k[i]) / len(test_ratings))

    return hits, ndcgs
