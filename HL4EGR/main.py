import argparse
import time
import os
import torch
from datetime import datetime

import model
import metrics
import dataloader as dataloader
from torch.utils.data import DataLoader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="weeplaces")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--ssl", type=float, default=0.3)
parser.add_argument("--sim", type=str, default="cos")
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--lambda_mi", type=float, default=1)
parser.add_argument("--drop_ratio", type=float, default=0.4)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epoch", default=50, type=int)
parser.add_argument("--pretrain_epoch", default=10, type=int)

parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--topK", type=list, default=[10,20,50])

args = parser.parse_args()
print('= ' * 20)
print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print(args)
print()

dataset = dataloader.GroupDataset(dataset=args.dataset)
gi_dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False)
device = torch.device(args.device)

rec_model = model.HCLGR(args.batch_size, dataset.num_items, dataset.num_users, args.emb_dim, args.layers, args.ssl, args.sim, args.tau, drop_ratio=args.drop_ratio, lambda_mi=args.lambda_mi, ui_hyper_graph=dataset.user_item_hyper_graph.to(device),
                        gu_hyper_graph=dataset.group_user_hyper_graph.to(device), gg_sim_item_hg= dataset.group_group_sim_item.to(device), gg_sim_prefer_hg = dataset.group_group_sim_prefer.to(device), device=device)
rec_model.to(device)

optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr, weight_decay=args.wd)

# Pretrain User-Item
print("Pre-training model on user-item interactions...")
optimizer_ui = torch.optim.Adam(rec_model.parameters(), lr=0.01, weight_decay=args.wd)
for epoch in range(args.pretrain_epoch):
    rec_model.train()

    user_train_data = dataset.user_pretrain_dataloader(args.batch_size)
    train_ui_loss = 0.0
    start_time = time.time()
    for _,(user_id,user_items) in enumerate(user_train_data):
        user_items = user_items.to(device)
        user_logits,_ = rec_model.user_item_hgcn.user_forward(user_id)
        user_loss = rec_model.user_loss(user_logits, user_items)

        optimizer_ui.zero_grad()
        user_loss.backward()
        optimizer_ui.step()
        train_ui_loss += user_loss.item()
    elapsed = time.time() - start_time
    print(f"[Epoch {epoch+1}] UI, time {elapsed:.2f}s, loss {train_ui_loss/len(user_train_data):.4f}")
    hits, ndcgs = metrics.user_model_leave_one_test(rec_model, dataset, dataset.user_test_ratings, device, k_list=args.topK)
    print(f"[Epoch {epoch+1}] User, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
print("Pre-train finish!\n")

_,all_user_emb = rec_model.user_item_hgcn.user_forward(list(range(dataset.num_users)))
all_group_users,all_group_mask = dataset.group_member_msg()
all_group_embed = []
for group_id in range(dataset.num_groups):
    user_pref_embeds = all_user_emb[all_group_users[group_id]].unsqueeze(0).to(device)
    group_mask = all_group_mask[group_id].unsqueeze(0).to(device)
    group_embed = rec_model.preference_aggregator(user_pref_embeds,group_mask)
    all_group_embed.append(group_embed)
all_group_embeds = torch.stack(all_group_embed).squeeze(1)

rec_model.group_predictor.weight.data = rec_model.user_item_hgcn.user_predictor.weight.data
for epoch in range(args.pretrain_epoch):
    rec_model.train()
    mi_epoch_loss = 0.0
    mi_epoch_start = time.time()
    print("finish data")
    for _,gi_data in enumerate(gi_dataloader):
        group_id,group_users,corrupted_user_id, group_masks, user_items, corrupt_user_items, _ = gi_data
        group_users = group_users.to(device)
        group_masks = group_masks.to(device)
        user_items = user_items.to(device)
        corrupt_user_items = corrupt_user_items.to(device)
        all_user_emb = all_user_emb.detach()
        all_group_embeds = all_group_embeds.detach()
        _, group_embeds, _,_,_,_,_,_ = rec_model(all_user_emb,all_group_embeds,group_id,group_users,group_masks)
        _,obs_user_embed = rec_model.user_item_hgcn.user_forward(group_users)
        obs_user_embed = obs_user_embed.detach()
        _,corrupt_user_embed = rec_model.user_item_hgcn.user_forward(corrupted_user_id)
        corrupt_user_embed = corrupt_user_embed.detach()

        score_obs = rec_model.discriminator(group_embeds, obs_user_embed)
        score_corrupt = rec_model.discriminator(group_embeds, corrupt_user_embed)

        mi_loss = rec_model.discriminator.mi_loss(score_obs, group_masks, score_corrupt, device=device)
        optimizer.zero_grad()
        mi_loss.backward()
        optimizer.step()
        mi_epoch_loss += mi_loss.item()
    elapsed = time.time() - mi_epoch_start
    print(f"[Epoch {epoch + 1}] MI, time {elapsed:.2f}s, loss {mi_epoch_loss / len(gi_data):.4f}")

print("Training model on group-item interactions...")
for epoch in range(args.epoch):
    epoch_start_time = time.time()
    rec_model.train()
    train_epoch_loss = 0.0
    g_epoch_loss, mi_epoch_loss, ug_epoch_loss = 0.0, 0.0, 0.0

    for _,gi_data in enumerate(gi_dataloader):
        group_id, group_users,corrupted_user_id, group_masks, user_items, corrupt_user_items, group_items = gi_data
        group_users = group_users.to(device)
        group_masks = group_masks.to(device)
        user_items = user_items.to(device)
        corrupt_user_items = corrupt_user_items.to(device)
        group_items = group_items.to(device)
        group_logits, group_embeds, scores_ug, group_embed_gg_sim_item_hgcn, group_embed_gg_sim_prefer_hgcn, agg_coef, hyper_coef, gg_sim_user_coef = rec_model(all_user_emb, all_group_embeds, group_id, group_users, group_masks)
        group_loss,g_loss,mi_loss,ug_loss = rec_model.loss(group_logits, group_embeds, scores_ug, group_masks, group_items, user_items,
                                                           corrupt_user_items, group_users, corrupted_user_id, group_embed_gg_sim_item_hgcn, group_embed_gg_sim_prefer_hgcn, device)
        optimizer.zero_grad()
        group_loss.backward()
        optimizer.step()
        train_epoch_loss += group_loss.item()
        g_epoch_loss += g_loss.item()
        mi_epoch_loss += mi_loss.item()
        ug_epoch_loss += ug_loss.item()
    elapsed = time.time() - epoch_start_time
    print(f"[Epoch {epoch+1}] GI, time {elapsed:.2f}s group-item loss: {train_epoch_loss/len(gi_data):.5f} \n g_loss: {g_epoch_loss/len(gi_data):.5f} mi_loss: {mi_epoch_loss/len(gi_data):.5f} ug_loss: {ug_epoch_loss/len(gi_data):.5f}")
    hits, ndcgs = metrics.group_model_leave_one_test(rec_model, dataset, dataset.group_test_ratings,all_user_emb,all_group_embeds, epoch,device, k_list=args.topK)
    print(f"[Epoch {epoch + 1}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
hits, ndcgs = metrics.group_model_leave_one_test(rec_model, dataset, dataset.group_test_ratings,all_user_emb,all_group_embeds,50, device, k_list=args.topK)
print(f"[Epoch { 1}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
print()
print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print('= ' * 20)
print("Done!")
