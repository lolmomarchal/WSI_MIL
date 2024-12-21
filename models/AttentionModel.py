import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Regular Attention Model
class AttentionModel(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.25, classes=1):
        super(AttentionModel, self).__init__()
        self.fc_linear = nn.Linear(input_dim, hidden_dim1)
        self.relu_activation = nn.ReLU()
        self.attention = nn.Linear(hidden_dim1, hidden_dim2)
        self.attention_c = nn.Linear(hidden_dim2, classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.fc_linear(x)
        # print(f"first linear layer: {x.shape} ")
        x = self.relu_activation(x)
        x = self.dropout(x)
        A = self.tanh(self.attention(x))
        A = self.dropout(A)
        Y_prob = self.attention_c(A)
        return Y_prob, x

# Gated Attention Model
class GatedAttentionModel(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.25, classes=1):
        super(GatedAttentionModel, self).__init__()
        self.fc_linear = nn.Linear(input_dim, hidden_dim1)
        self.relu_activation = nn.ReLU()
        self.attention = nn.Linear(hidden_dim1, hidden_dim2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.attention_c = nn.Linear(hidden_dim2, classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        # print("Gated Attention")
        # print(f"original input: {x.shape}")
        x = self.fc_linear(x)
        # print(f"first linear layer: {x.shape} ")
        x = self.relu_activation(x)
        x = self.dropout(x)
        # print(f"after relu and dropout layer: {x.shape} ")
        A = self.sigmoid(self.attention(x))
        # print(f" sigmoid layer after linear layer: {A.shape}")
        A = self.dropout(A)
        B = self.tanh(self.attention(x))
        # print(f" tanh layer after linear layer: {B.shape}")
        B = self.dropout(B)
        # final attention scores
        Y_prob = self.attention_c(A.mul(B))
        # print(f" final attention scores shape : {Y_prob.shape}")
        return Y_prob, x
