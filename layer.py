import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from params import settings
args = settings()

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False) # Score each element
        )

    def forward(self, z):
        w = self.project(z)
        beta = th.softmax(w, dim=1)
        return (beta * z).sum(1), beta # Weighted sum and attention weights

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return th.mean(seq, 0)
        else:
            msk = th.unsqueeze(msk, -1)
            return th.sum(seq * msk, 0) / th.sum(msk)

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()
        sc_1 = self.fn(h1, c_x1).squeeze(1)
        sc_2 = self.fn(h2, c_x2).squeeze(1)
        sc_3 = self.fn(h3, c_x1).squeeze(1)
        sc_4 = self.fn(h4, c_x2).squeeze(1)
        logits = th.cat((sc_1, sc_2, sc_3, sc_4))
        return logits

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.prelu1 = nn.PReLU(nhid)
        self.gc2 = GCNConv(nhid, out)
        self.prelu2 = nn.PReLU(out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.prelu1(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.prelu2(self.gc2(x, adj))
        return x

class SSCLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, decoder1):
        super(SSCLP, self).__init__()
        self.encoder1 = GCN(in_dim, hid_dim, out_dim)
        self.encoder2 = GCN(in_dim, hid_dim, out_dim)
        self.encoder3 = GCN(in_dim, hid_dim, out_dim)
        self.encoder4 = GCN(in_dim, hid_dim, out_dim)
        self.pooling = AvgReadout()
        self.attention = Attention(out_dim)
        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()
        self.local_mlp = nn.Linear(out_dim, out_dim)
        self.global_mlp = nn.Linear(out_dim, out_dim)
        self.decoder1 = nn.Linear(out_dim * 4, decoder1)
        self.decoder2 = nn.Linear(decoder1, 1)

    def forward(self, data_s, data_f, idx, perturbation_strength=0.1):
        feat, s_graph = data_s.x, data_s.edge_index
        shuff_feat, f_graph = data_f.x, data_f.edge_index

        # Add Gaussian perturbation to shuffled features
        noise = th.randn_like(shuff_feat) * perturbation_strength
        perturbed_feat = shuff_feat + noise

        h1 = self.encoder1(feat, s_graph)
        h2 = self.encoder2(feat, f_graph)
        h1 = self.local_mlp(h1)
        h2 = self.local_mlp(h2)

        h3 = self.encoder1(perturbed_feat, s_graph)
        h4 = self.encoder2(perturbed_feat, f_graph)
        h3 = self.local_mlp(h3)
        h4 = self.local_mlp(h4)

        h5 = self.encoder3(feat, s_graph)
        h6 = self.encoder3(feat, f_graph)

        c1 = self.act_fn(self.global_mlp(self.pooling(h1)))
        c2 = self.act_fn(self.global_mlp(self.pooling(h2)))

        out = self.disc(h1, h2, h3, h4, c1, c2)
        h_com = (h5 + h6) / 2
        emb = th.stack([h1, h2, h_com], dim=1)
        emb, att = self.attention(emb)

        if args.task_type == 'LDA':
            entity1 = emb[idx[0]]
            entity2 = emb[idx[1] + 386]
        elif args.task_type == 'MDA':
            entity1 = emb[idx[0] + 702]
            entity2 = emb[idx[1] + 386]
        elif args.task_type == 'LMI':
            entity1 = emb[idx[0]]
            entity2 = emb[idx[1] + 702]

        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = th.cat((entity1, entity2), dim=1)
        feature = th.cat((add, product, concatenate), dim=1)

        log1 = F.relu(self.decoder1(feature))
        log = self.decoder2(log1)
        return out, log, h1, h3  # return clean & perturbed embeddings too
