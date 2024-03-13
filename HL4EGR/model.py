import torch
import torch.nn as nn
import torch.nn.functional as F

class HGBCN(nn.Module):
    def __init__(self, dim):
        super(HGBCN, self).__init__()
        self.aggregation = nn.Linear(2 * dim, dim)

    def forward(self, user_emb, item_emb, hyper_graph):
        node_msg = torch.sparse.mm(hyper_graph, item_emb)
        edge_node_element = node_msg * user_emb
        msg = self.aggregation(torch.cat([node_msg, edge_node_element], dim=1))
        norm_emb = torch.mm(hyper_graph.t(), msg)
        return norm_emb, msg


class HGBCN_2(nn.Module):
    def __init__(self, dim):
        super(HGBCN_2, self).__init__()
        self.aggregation = nn.Linear(2 * dim, dim)

    def forward(self, user_emb, item_emb, hyper_graph):
        node_msg = torch.sparse.mm(hyper_graph, item_emb)
        edge_node_element = node_msg * user_emb
        msg = self.aggregation(torch.cat([node_msg, edge_node_element], dim=1))
        norm_emb = torch.mm(hyper_graph.t(), msg)
        return norm_emb, msg


class HGBCN_3(nn.Module):
    def __init__(self, dim):
        super(HGBCN_3, self).__init__()
        self.aggregation = nn.Linear(2 * dim, dim)

    def forward(self, user_emb, item_emb, hyper_graph):
        node_msg = torch.sparse.mm(hyper_graph, item_emb)
        edge_node_element = node_msg * user_emb
        msg = self.aggregation(torch.cat([node_msg, edge_node_element], dim=1))
        norm_emb = torch.mm(hyper_graph.t(), msg)
        return norm_emb, msg


class HGBCN_4(nn.Module):
    def __init__(self, dim):
        super(HGBCN_4, self).__init__()
        self.aggregation = nn.Linear(2 * dim, dim)

    def forward(self, user_emb, item_emb, hyper_graph):
        node_msg = torch.sparse.mm(hyper_graph, item_emb)
        edge_node_element = node_msg * user_emb
        msg = self.aggregation(torch.cat([node_msg, edge_node_element], dim=1))
        norm_emb = torch.mm(hyper_graph.t(), msg)
        return norm_emb, msg


class HGCN_UI(nn.Module):
    def __init__(self, hyper_graph, layers, dim, n_items, n_users, device):
        super(HGCN_UI, self).__init__()
        self.layers = layers
        self.hyper_graph = hyper_graph
        self.hgnn = [HGBCN(dim).to(device) for _ in range(layers)]
        self.n_items = n_items
        self.n_users = n_users

        self.user_predictor = nn.Linear(dim, n_items, bias=False)
        nn.init.xavier_uniform_(self.user_predictor.weight)

        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_emb, item_emb, num_users, num_items):
        final_node = [item_emb]
        final_edge = [user_emb]
        for i in range(self.layers):
            hgnn = self.hgnn[i]
            item_emb, user_emb = hgnn(user_emb, item_emb, self.hyper_graph)
            final_node.append(item_emb)
            final_edge.append(user_emb)
        final_node_emb = torch.sum(torch.stack(final_node), dim=0)
        final_edge_emb = torch.sum(torch.stack(final_edge), dim=0)
        return final_node_emb, final_edge_emb

    def user_forward(self, user_id):
        _, users_emb = self.forward(self.user_embedding.weight, self.item_embedding.weight, self.n_users, self.n_items)
        user_emb = users_emb[user_id]
        logits = self.user_predictor(user_emb)
        return logits, user_emb


class HGCN_GU(nn.Module):
    def __init__(self, hyper_graph, layers, dim, device):
        super(HGCN_GU, self).__init__()
        self.layers = layers
        self.hyper_graph = hyper_graph
        self.hgnn = [HGBCN_2(dim).to(device) for _ in range(layers)]

    def forward(self, group_emb, user_emb):
        final_node = [user_emb]
        final_edge = [group_emb]
        for i in range(self.layers):
            hgnn = self.hgnn[i]
            user_emb, group_emb = hgnn(group_emb, user_emb, self.hyper_graph)
            final_node.append(user_emb)
            final_edge.append(group_emb)
        final_node_emb = torch.sum(torch.stack(final_node), dim=0)
        final_edge_emb = torch.sum(torch.stack(final_edge), dim=0)
        return final_node_emb, final_edge_emb

    def group_forward(self, all_user_emb, all_group_emb, group_id):
        _, groups_emb = self.forward(all_group_emb, all_user_emb)
        group_emb = groups_emb[group_id]
        return group_emb


class HGCN_GG_ITEM(nn.Module):
    def __init__(self, hyper_graph, layers, dim, device):
        super(HGCN_GG_ITEM, self).__init__()
        self.layers = layers
        self.hyper_graph = hyper_graph
        self.hgnn = [HGBCN_3(dim).to(device) for _ in range(layers)]

    def forward(self, group_emb, user_emb):
        final_node = [user_emb]
        final_edge = [group_emb]
        for i in range(self.layers):
            hgnn = self.hgnn[i]
            user_emb, group_emb = hgnn(group_emb, user_emb, self.hyper_graph)
            final_node.append(user_emb)
            final_edge.append(group_emb)
        final_node_emb = torch.sum(torch.stack(final_node), dim=0)
        final_edge_emb = torch.sum(torch.stack(final_edge), dim=0)
        return final_node_emb, final_edge_emb

    def group_forward(self, all_group_emb, group_id):
        _, groups_emb = self.forward(all_group_emb, all_group_emb)
        group_emb = groups_emb[group_id]
        return group_emb


class HGCN_GG_PREFER(nn.Module):
    def __init__(self, hyper_graph, layers, dim, device):
        super(HGCN_GG_PREFER, self).__init__()
        self.layers = layers
        self.hyper_graph = hyper_graph
        self.hgnn = [HGBCN_4(dim).to(device) for _ in range(layers)]

    def forward(self, group_emb, user_emb):
        final_node = [user_emb]
        final_edge = [group_emb]
        for i in range(self.layers):
            hgnn = self.hgnn[i]
            user_emb, group_emb = hgnn(group_emb, user_emb, self.hyper_graph)
            final_node.append(user_emb)
            final_edge.append(group_emb)
        final_node_emb = torch.sum(torch.stack(final_node), dim=0)
        final_edge_emb = torch.sum(torch.stack(final_edge), dim=0)
        return final_node_emb, final_edge_emb

    def group_forward(self, all_group_emb, group_id):
        _, groups_emb = self.forward(all_group_emb, all_group_emb)
        group_emb = groups_emb[group_id]
        return group_emb


class HCLGR(nn.Module):
    def __init__(self, batch_size, n_items, n_users, emb_dim, layers, ssl, sim, tau, ui_hyper_graph, gu_hyper_graph,
                 gg_sim_item_hg, gg_sim_prefer_hg, device, lambda_mi=0.1, drop_ratio=0.4):
        super(HCLGR, self).__init__()
        self.n_items = n_items
        self.lambda_mi = lambda_mi
        self.drop = nn.Dropout(drop_ratio)
        self.n_users = n_users
        self.emb_dim = emb_dim
        self.layers = layers
        self.ssl = ssl
        self.sim = sim
        self.tau = tau
        self.batch_size = batch_size

        self.user_item_hgcn = HGCN_UI(ui_hyper_graph, self.layers, self.emb_dim, self.n_items, n_users, device)
        self.group_user_hgcn = HGCN_GU(gu_hyper_graph, self.layers, self.emb_dim, device)
        self.gg_sim_item_hgcn = HGCN_GG_ITEM(gg_sim_item_hg, self.layers, self.emb_dim, device)
        self.gg_sim_prefer_hgcn = HGCN_GG_PREFER(gg_sim_prefer_hg, self.layers, self.emb_dim, device)
        self.preference_aggregator = AttentionAggregator(self.emb_dim)
        self.group_predictor = nn.Linear(self.emb_dim, self.n_items, bias=False)
        nn.init.xavier_uniform_(self.group_predictor.weight)

        # 自适应融合
        self.agg_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.gg_sim_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        self.discriminator = Discriminator(emb_dim=self.emb_dim)
        # 对比学习
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, all_user_emb, all_group_emb, group_id, user_id, group_mask):

        # G-U HGCN
        group_embed_gu_hgcn = self.group_user_hgcn.group_forward(all_user_emb, all_group_emb, group_id)
        # G-G sim user HGCN
        group_embed_gg_sim_item_hgcn = self.gg_sim_item_hgcn.group_forward(all_group_emb, group_id)
        # G-G sim item HGCN
        group_embed_gg_sim_prefer_hgcn = self.gg_sim_prefer_hgcn.group_forward(all_group_emb, group_id)
        # user agg 聚合用u-i超图得到的embedding
        _, user_pref_embeds = self.user_item_hgcn.user_forward(user_id)
        group_embed_user_agg = self.preference_aggregator(user_pref_embeds, group_mask)
        # 自适应融合
        agg_coef, hyper_coef, gg_sim_coef = self.agg_gate(group_embed_user_agg), self.hyper_gate(
            group_embed_gu_hgcn), self.gg_sim_gate(group_embed_gg_sim_prefer_hgcn)
        group_embed = agg_coef * group_embed_user_agg + hyper_coef * group_embed_gu_hgcn + gg_sim_coef * group_embed_gg_sim_prefer_hgcn

        group_logit = self.group_predictor(group_embed)

        if self.train:
            _, obs_user_embeds = self.user_item_hgcn.user_forward(user_id)
            scores_ug = self.discriminator(group_embed, obs_user_embeds).detach()
            return group_logit, group_embed, scores_ug, group_embed_gg_sim_item_hgcn, group_embed_gg_sim_prefer_hgcn, agg_coef, hyper_coef, gg_sim_coef
        else:
            return group_logit, group_embed, group_embed_gg_sim_item_hgcn, group_embed_gg_sim_prefer_hgcn, agg_coef, hyper_coef, gg_sim_coef

    def multinomial_loss(self, logits, items):
        return -torch.mean(torch.sum(F.log_softmax(logits, 1) * items, -1))

    def user_loss(self, user_logits, user_items):
        return self.multinomial_loss(user_logits, user_items)

    def info_max_group_loss(self, group_logits, group_embeds, scores_ug, group_mask, group_items, user_items,
                            corrupted_user_items, user_id, corrupted_user_id, device="cpu"):
        _, group_user_embeds = self.user_item_hgcn.user_forward(user_id)
        _, corrupt_user_embeds = self.user_item_hgcn.user_forward(corrupted_user_id)

        scores_observed = self.discriminator(group_embeds, group_user_embeds)  # [B, G]
        scores_corrupted = self.discriminator(group_embeds, corrupt_user_embeds)  # [B, N]

        mi_loss = self.discriminator.mi_loss(scores_observed, group_mask, scores_corrupted, device=device)

        ui_sum = user_items.sum(2, keepdim=True)  # [B, G]
        user_items_norm = user_items / torch.max(torch.ones_like(ui_sum), ui_sum)  # [B, G, I]
        gi_sum = group_items.sum(1, keepdim=True)
        group_items_norm = group_items / torch.max(torch.ones_like(gi_sum), gi_sum)  # [B, I]
        assert scores_ug.requires_grad is False

        group_mask_zeros = torch.exp(group_mask).unsqueeze(2)  # [B, G, 1]
        scores_ug = torch.sigmoid(scores_ug)  # [B, G, 1]

        user_items_norm = torch.sum(user_items_norm * scores_ug * group_mask_zeros, dim=1) / group_mask_zeros.sum(1)
        user_group_loss = self.multinomial_loss(group_logits, user_items_norm)
        group_loss = self.multinomial_loss(group_logits, group_items_norm)

        return mi_loss, user_group_loss, group_loss

    def loss(self, group_logits, summary_embeds, scores_ug, group_mask, group_items, user_items, corrupted_user_items,
             user_id, corrupted_user_id,
             group_embed_gg_sim_item_hgcn, group_embed_gg_sim_prefer_hgcn, device='cpu'):
        mi_loss, user_group_loss, group_loss = self.info_max_group_loss(group_logits, summary_embeds, scores_ug,
                                                                        group_mask, group_items, user_items,
                                                                        corrupted_user_items, user_id,
                                                                        corrupted_user_id, device)
        if self.ssl != 0:
            nce_logits, nce_labels = self.info_nce(group_embed_gg_sim_item_hgcn, group_embed_gg_sim_prefer_hgcn,
                                                   temp=self.tau,
                                                   batch_size=group_embed_gg_sim_item_hgcn.shape[0], sim=self.sim)
            ssl_loss = self.aug_nce_fct(nce_logits, nce_labels)
        else:
            ssl_loss = 0

        return group_loss + mi_loss + self.lambda_mi * user_group_loss + self.ssl * ssl_loss, group_loss, mi_loss, user_group_loss

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            # print(1)
            mask = self.mask_correlated_samples(batch_size)
        else:
            # print(0)
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


class AttentionAggregator(nn.Module):

    def __init__(self, output_dim, drop_ratio=0.):
        super(AttentionAggregator, self).__init__()

        self.attention = nn.Linear(output_dim, 1)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x, mask):
        attention_out = torch.tanh(self.attention(x))
        if mask is None:
            weight = torch.softmax(attention_out, dim=1)
        else:
            weight = torch.softmax(attention_out + mask.unsqueeze(2), dim=1)
        ret = torch.matmul(x.transpose(2, 1), weight).squeeze(2)
        return ret


class Discriminator(nn.Module):

    def __init__(self, emb_dim=64):
        super(Discriminator, self).__init__()
        self.emb_dim = emb_dim

        self.fc_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)

        self.bilinear_layer = nn.Bilinear(self.emb_dim, self.emb_dim, 1)
        nn.init.xavier_uniform_(self.bilinear_layer.weight)
        nn.init.zeros_(self.bilinear_layer.bias)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, group_inputs, user_inputs):
        group_emb = torch.tanh(self.fc_layer(group_inputs))

        user_emb = torch.tanh(self.fc_layer(user_inputs))

        return self.bilinear_layer(user_emb, group_emb.unsqueeze(1).repeat(1, user_inputs.shape[1], 1))

    def mi_loss(self, scores_group, group_mask, scores_corrupted, device="cpu"):
        batch_size = scores_group.shape[0]

        pos_size, neg_size = scores_group.shape[1], scores_corrupted.shape[1]

        one_labels = torch.ones(batch_size, pos_size).to(device)
        zero_labels = torch.zeros(batch_size, neg_size).to(device)

        labels = torch.cat((one_labels, zero_labels), 1)
        logits = torch.cat((scores_group, scores_corrupted), 1).squeeze(2)

        mask = torch.cat((torch.exp(group_mask), torch.ones([batch_size, neg_size]).to(device)), 1)

        mi_loss = self.bce_loss(logits * mask, labels * mask) * (batch_size * (pos_size + neg_size)) \
                  / (torch.exp(group_mask).sum() + batch_size * neg_size)

        return mi_loss