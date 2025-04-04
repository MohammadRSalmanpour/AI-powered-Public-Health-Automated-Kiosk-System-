
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
from torch_geometric.nn import GATConv
from torch.nn import MultiheadAttention
from torch.utils.data import DataLoader, TensorDataset



class Kiosk(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'), ddi_in_memory=True):
        super(Kiosk, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # GCN for EHR graph
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)

        # GAT for DDI graph
        self.ddi_gat = GATConv(emb_dim, emb_dim, heads=2, dropout=0.5)  

        # Multihead Attention for feature integration
        self.num_heads = 2
        self.cross_attn = MultiheadAttention(emb_dim, self.num_heads)  # Cross Attention

        # Add the 'inter' attribute
        self.inter = nn.Parameter(torch.FloatTensor(1))  # Interaction parameter

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()
  
    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:
            
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        ) # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](
            i2_seq
        )
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)

 
            
            # Graph memory module
        '''I:generate current input'''
        query = queries[-1:]  # (1, dim)
    
        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            # Convert DDI adjacency matrix to edge index format for GAT
            ddi_edge_index = self.tensor_ddi_adj.nonzero().t().contiguous().to(self.device)
            
            # Apply GAT to DDI graph
            ddi_embeddings = self.ddi_gat(torch.eye(self.vocab_size[2]).to(self.device), ddi_edge_index)
            
            # Combine EHR and DDI embeddings using 'self.inter'
            drug_memory = self.ehr_gcn() - ddi_embeddings * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()
    
        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)]  # (seq-1, dim)
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device)  # (seq-1, size)
    
        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)
    
        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()))  # (1, seq-1)        
            weighted_values = visit_weight.mm(history_values)  # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory)  # (1, dim)
        else:
            fact2 = fact1
     
       # print("query shape:", query.shape)
       # print("fact1 shape:", fact1.shape)   
       # print("fact2 shape:", fact2.shape)
        
        # Apply Multihead Cross Attention
        cross_attn_input = torch.cat([query, fact1, fact2], dim=0)  # (3, dim)
        cross_attn_output, cross_attn_weights = self.cross_attn(cross_attn_input, cross_attn_input, cross_attn_input)
        #print("cross_attn_output shape:", cross_attn_output.shape)
        # Reshape cross_attn_output to (1, dim)
        cross_attn_output = cross_attn_output.mean(dim=0).unsqueeze(0)  
        
        
        
        # Reshape cross_attn_output to (1, dim)
        
       # print("cross_attn_output2 Reshape:", cross_attn_output.shape)
        # Combine embeddings
        combined_output = torch.cat([query, fact1, fact2, cross_attn_output], dim=-1)  # (1, dim * 4)
        output = self.output(combined_output)  # (1, vocab_size[2])
       # print("foutput shape:", output.shape)
        
        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights.""" 
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
    
        # Initialize 'inter'
        self.inter.data.uniform_(-initrange, initrange)


########################################################################

###############################################


###############################################
