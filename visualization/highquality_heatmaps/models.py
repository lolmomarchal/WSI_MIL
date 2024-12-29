import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import torch.optim as optim
import numpy as np
import warnings
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

warnings.filterwarnings("ignore")
# =========================== Feature Extraction ===============================
def encoder(device='cpu'):
    encoder_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-3])
    encoder_model.eval()
    if device == 'cpu':
        encoder_model = torch.quantization.quantize_dynamic(encoder_model, {torch.nn.Linear}, dtype=torch.qint8)
    encoder_model.to(device)
    return encoder_model


# =========================== Models ===========================================
# high quality scores
class Scores(nn.Module):
    def __init__(self, mil):
        super(Scores, self).__init__()
        self.mil = mil

    def forward(self, h):
        attention, h = self.mil.attention_net(h)
        logits = self.mil.classifiers(h)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


# high quality gated-attention
class gated_att(nn.Module):
    def __init__(self, mil):
        super(gated_att, self).__init__()
        self.encoder = mil.attention_net.fc_linear
        self.relu = nn.ReLU()
        self.dropout = mil.attention_net.dropout
        self.attention = mil.attention_net.attention
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.attention_c = mil.attention_net.attention_c

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.dropout(x)
        A = self.sigmoid(self.attention(x))
        A = self.dropout(A)
        B = self.tanh(self.attention(x))
        B = self.dropout(B)
        # final attention scores
        Y_prob = self.attention_c(A.mul(B))
        return Y_prob


# high quality self attention
class att(nn.Module):
    def __init__(self, mil):
        super(att, self).__init__()
        self.encoder = mil.attention_net.fc_linear
        self.relu = nn.ReLU()
        self.dropout = mil.attention_net.dropout
        self.attention = mil.attention_net.attention
        self.tanh = nn.Tanh()
        self.attention_c = mil.attention_net.attention_c

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.dropout(x)
        B = self.tanh(self.attention(x))
        B = self.dropout(B)
        # final attention scores
        Y_prob = self.attention_c(B)
        return Y_prob


# =========================== Converting ====================================

# converting 1d to 2d for high quality
def dropout1d_to_dropout2d(dropout1d):
    """ converts dropout1d to dropout2d layer"""
    return nn.Dropout2d(dropout1d.p)


def linear_to_conv2d(linear):
    """Converts a fully connected layer to a 1x1 Conv2d layer with the same weights."""
    conv = nn.Conv2d(
        in_channels=linear.in_features, out_channels=linear.out_features, kernel_size=1
    )
    conv.load_state_dict(
        {
            "weight": linear.weight.view(conv.weight.shape),
            "bias": linear.bias.view(conv.bias.shape),
        }
    )
    return conv


def high_quality(instance, model_, weights_path, attention="gated", device="cpu", previously_encoded=False):
    # load model weights & set model to eval
    model = type(model_)()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # convert linear layers to conv and dropout to dropout2d

    # encoder
    model.attention_net.fc_linear = linear_to_conv2d(model.attention_net.fc_linear)
    # attention
    model.attention_net.attention = linear_to_conv2d(model.attention_net.attention)
    model.attention_net.attention_c = linear_to_conv2d(model.attention_net.attention_c)
    model.attention_net.dropout = dropout1d_to_dropout2d(model.attention_net.dropout)
    # classifiers
    model.classifiers = linear_to_conv2d(model.classifiers)

    # if not encoded without average pool + flattening, get encoding
    if not previously_encoded:
        en = encoder(device=device)
        tile = PIL.Image.open(instance)
        transform = transforms.ToTensor()
        tile_tensor = transform(tile).unsqueeze(0).to(device)
        with torch.no_grad():
            encoded_fet = en(tile_tensor).cpu()
    else:  # already with proper
        encoded_fet = instance

    scores = Scores(model).eval().to(device)
    if attention == "gated":
        attn = gated_att(model).eval().to(device)
    else:
        attn = att(model).eval().to(device)

    # high quality scores
    with torch.no_grad():
        high_attention = attn(encoded_fet).cpu()
        high_scores = scores(encoded_fet).cpu()
    return high_attention, high_scores