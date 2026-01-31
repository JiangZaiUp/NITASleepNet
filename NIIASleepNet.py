


import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, cohen_kappa_score, precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import plotly.express as px

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.model_selection import KFold
from datetime import datetime
import sys
import torch.nn.functional as F
import torchmetrics.functional as F_metrics 
import multiprocessing
import gc
import psutil
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ISATSleepNet Training Configuration')
    parser.add_argument('--log_dir', type=str, required=True, 
                        help='Directory to save training logs')
    parser.add_argument('--data_folder_train', type=str, required=True, 
                        help='Directory of training data')
    parser.add_argument('--data_folder_test', type=str, required=True, 
                        help='Directory of test/validation data')
    parser.add_argument('--fold_dir', type=str, required=True, 
                        help='Directory to save trained models and training plots')
    parser.add_argument('--start_fold', type=int, default=0, 
                        help='Starting fold index (0 for new training, default: 0)')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0], 
                        help='GPU indices to use (e.g., --device_ids 0 1, default: [0])')
    parser.add_argument('--num_epoch', type=int, nargs='+', default=100, 
                        help='Train epoch number')
    
    args = parser.parse_args()
    return args

args = parse_args()

log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "training_log.txt")
log_file = open(log_file_path, 'a', encoding='utf-8')


class Logger(object):
    def __init__(self, file, terminal):
        self.file = file
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        pass


sys.stdout = Logger(log_file, sys.stdout)
sys.stderr = Logger(log_file, sys.stderr)




class EEGSleepDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)  
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]





class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=5, class_weights=None):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)  
        
    def forward(self, logits, targets):
        
        ce = self.ce_loss(logits, targets)
        
        
        probs = F.softmax(logits, dim=1)
        y_true_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        
        tp = torch.sum(y_true_onehot * probs, dim=0)
        fp = torch.sum((1 - y_true_onehot) * probs, dim=0)
        fn = torch.sum(y_true_onehot * (1 - probs), dim=0)
        
        
        f1_per_class = 2 * tp / (2 * tp + fp + fn + 1e-8)
        f1_loss = 1 - torch.mean(f1_per_class)  
        
        
        return self.alpha * ce + (1 - self.alpha) * f1_loss

def compute_class_weights(y_train, device='cpu', smoothing=1.0):
    
    
    unique_labels, label_counts = np.unique(y_train, return_counts=True)
    

    class_counts = np.bincount(y_train)
    total_samples = len(y_train)

    
    
    class_weights = total_samples / (class_counts + smoothing)
    class_weights[1] *= 2.5  
    class_weights = class_weights / np.mean(class_weights)
    class_weights = total_samples / (class_counts + smoothing)

    
    class_weights = class_weights / np.mean(class_weights)

    return torch.tensor(class_weights, dtype=torch.float32).to(device)




class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        q1 = self.query(x1)
        k2 = self.key(x2)
        v2 = self.value(x2)

        attn_scores = torch.matmul(q1, k2.transpose(-2, -1)) / np.sqrt(q1.size(-1))
        attn_probs = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_probs, v2)

        return attn_output

class LightChannelScale(nn.Module):
    def __init__(self, channels, eps=0.1):
        
        super().__init__()
        self.p = nn.Parameter(torch.zeros(channels))  
        self.eps = float(eps)

    def forward(self, x):
        
        scale = 1.0 + self.eps * torch.tanh(self.p)  
        scale = scale.view(1, -1, 1).to(x.device).to(x.dtype)
        return x * scale

class EOGModulation(nn.Module):
    
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim * 2)  
        )

    def forward(self, eeg_feat, eog_feat):
        
        
        params = self.net(eog_feat)
        scale, gate = params.chunk(2, dim=-1)

        scale = torch.tanh(scale)        
        gate  = torch.sigmoid(gate)      

        
        eeg_mod = eeg_feat * (1.0 + 0.3 * scale) * gate
        return eeg_mod

class EEGCorticalFusion(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, eeg1, eeg2):
        return self.fuse(torch.cat([eeg1, eeg2], dim=-1))




class SleepAttractorReadout(nn.Module):
    
    def __init__(self, dim, num_classes):
        super().__init__()
        self.attractors = nn.Parameter(torch.randn(num_classes, dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, h):
        """
        h: (B,T,C)
        """
        s = h.mean(dim=1)  
        dist = torch.cdist(s, self.attractors)
        logits = -dist / self.temperature
        return logits

class AttractorStateRectifier(nn.Module):
    
    def __init__(self, dim, num_classes, beta=0.5):
        super().__init__()
        self.attractors = nn.Parameter(torch.randn(num_classes, dim))
        self.beta = beta
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, h):
        
        s = h.mean(dim=1)  

        
        sim = torch.matmul(
            F.normalize(s, dim=-1),
            F.normalize(self.attractors, dim=-1).T
        )  

        weights = F.softmax(sim / self.temperature, dim=-1)  

        a_hat = torch.matmul(weights, self.attractors)  

        s_refined = s + self.beta * (a_hat - s)

        return h + self.beta * (s_refined.unsqueeze(1) - h)



class SharedStateProjector(nn.Module):
    
    def __init__(self, in_dim, state_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, state_dim, bias=False),
            nn.LayerNorm(state_dim)
        )

    def forward(self, x):
        """
        x: (B, T, C)
        """
        return self.proj(x)

class CrossModalSimilarityGate(nn.Module):
   
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_state, y_state):
        
        x_n = F.normalize(x_state, dim=-1, eps=self.eps)
        y_n = F.normalize(y_state, dim=-1, eps=self.eps)

        sim = (x_n * y_n).sum(dim=-1, keepdim=True)   
        gate = (sim + 1.0) / 2.0                      

        return gate * self.scale


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, eps=0.1):
        super(MultiScaleBlock, self).__init__()

        
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=30, stride=5, padding=14)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=50, stride=10, padding=24)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=70, stride=20, padding=34)

        
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()

        
        self.scale = LightChannelScale(out_channels * 3, eps=eps)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
     
        out3 = self.conv3(x)  
        out5 = self.conv5(x)  
        out7 = self.conv7(x)  

        
        target_len = out3.size(2)
        out5 = F.adaptive_avg_pool1d(out5, target_len)
        out7 = F.adaptive_avg_pool1d(out7, target_len)

        
        out = torch.cat([out3, out5, out7], dim=1)

        
        out = self.bn(out)
        out = self.scale(out)   
        out = self.relu(out)
        out = self.dropout(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x


class EOGFeatureExtractor(nn.Module):
    
    def __init__(self, base_channels=64, dropout=0.5):
        super().__init__()
        half = base_channels // 2

        
        self.slow_branch = nn.Sequential(
            nn.Conv1d(1, half, kernel_size=101, stride=4, padding=50, bias=False),
            nn.BatchNorm1d(half),
            nn.ReLU(inplace=True),
        )
        
        self.fast_branch = nn.Sequential(
            nn.Conv1d(1, half, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(half),
            nn.ReLU(inplace=True),
        )

        
        self.res_block1 = self._res_block(base_channels, kernel_size=51, dilation=2)
        self.res_block2 = self._res_block(base_channels, kernel_size=51, dilation=4)

        
        self.fuse_conv = nn.Conv1d(base_channels * 3, 128, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm1d(128)
        self.fuse_act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _res_block(ch, kernel_size=51, dilation=1):
        pad = ((kernel_size - 1) // 2) * dilation
        return nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )

    def _diff(self, x):
        
        diff = x[:, :, 1:] - x[:, :, :-1]
        diff = nn.functional.pad(diff, (1, 0))
        return diff

    def forward(self, x, target_len):
       
        slow = self.slow_branch(x)   
        fast_in = self._diff(x)
        fast = self.fast_branch(fast_in)  

        
        Lm = min(slow.size(2), fast.size(2))
        slow = nn.functional.adaptive_avg_pool1d(slow, Lm)
        fast = nn.functional.adaptive_avg_pool1d(fast, Lm)

        feat = torch.cat([slow, fast], dim=1)  

        
        res1 = self.res_block1(feat)
        feat = nn.functional.relu(feat + res1, inplace=True)
        res2 = self.res_block2(feat)
        feat = nn.functional.relu(feat + res2, inplace=True)  

        
        p1 = nn.functional.adaptive_avg_pool1d(feat, target_len)                     
        p2 = nn.functional.adaptive_avg_pool1d(feat, max(target_len // 2, 1))        
        p2 = nn.functional.interpolate(p2, size=target_len, mode='linear', align_corners=False)
        p3 = nn.functional.adaptive_avg_pool1d(feat, max(target_len // 4, 1))        
        p3 = nn.functional.interpolate(p3, size=target_len, mode='linear', align_corners=False)

        spp = torch.cat([p1, p2, p3], dim=1)  

        
        out = self.fuse_conv(spp)              
        out = self.fuse_bn(out)
        out = self.fuse_act(out)
        out = self.drop(out)
        out = out.permute(0, 2, 1)            
        return out

class NIIASleepNet(nn.Module):
    def __init__(self, d_model=128, num_classes=5, lstm_layers=2, dropout=0.5):
        super(NIIASleepNet, self).__init__()

        self.eeg_cnn = nn.Sequential(
            MultiScaleBlock(1, 32),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout)
        )

        
        self.new_eeg_cnn = nn.Sequential(
            MultiScaleBlock(1, 32),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout)
        )

        
        self.eog_cnn = EOGFeatureExtractor(base_channels=64, dropout=dropout)

        
        self.state_proj_eeg = SharedStateProjector(128, 64)
        self.state_proj_new_eeg = SharedStateProjector(128, 64)
        self.state_proj_eog = SharedStateProjector(128, 64)

        
        self.gate_eeg_new = CrossModalSimilarityGate()
        self.gate_eeg_eog = CrossModalSimilarityGate()

        self.gate_new_eeg = CrossModalSimilarityGate()
        self.gate_new_eog = CrossModalSimilarityGate()

        self.gate_eog_eeg = CrossModalSimilarityGate()
        self.gate_eog_new = CrossModalSimilarityGate()

        
        self.cross_eeg2new_eeg = CrossAttention(128)
        self.cross_new_eeg2eeg = CrossAttention(128)
        self.cross_new_eeg2eog = CrossAttention(128)
        self.cross_eog2new_eeg = CrossAttention(128)

        
        self.cross_eeg2eog = CrossAttention(128)
        self.cross_eog2eeg = CrossAttention(128)

        
        self.fusion_proj = nn.Linear(128 * 3, d_model)

        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        self.attn = MultiHeadSelfAttention(d_model * 2, num_heads=4, dropout=dropout)

        self.attractor_rectifier = AttractorStateRectifier(
            dim=d_model * 2,
            num_classes=num_classes,
            beta=0.5
        )

        
        self.attractor_readout = SleepAttractorReadout(d_model * 2, num_classes)

        self.eeg_fusion = EEGCorticalFusion(128)
        self.eog_modulation = EOGModulation(128)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )


    def forward(self, x):
            
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)  

        self.lstm.flatten_parameters()
        
        eeg = x[:, :, 0].unsqueeze(1)
        eog = x[:, :, 1].unsqueeze(1)
        new_eeg = x[:, :, 2].unsqueeze(1)  

        eeg_feat = self.eeg_cnn(eeg).permute(0, 2, 1)
        new_eeg_feat = self.new_eeg_cnn(new_eeg).permute(0, 2, 1)
        
        
        target_len = eeg_feat.size(1)  
        eog_feat = self.eog_cnn(eog, target_len=target_len)  

        
        eeg_fused_new_eeg = self.cross_eeg2new_eeg(eeg_feat, new_eeg_feat)
        new_eeg_fused_eeg = self.cross_new_eeg2eeg(new_eeg_feat, eeg_feat)
        new_eeg_fused_eog = self.cross_new_eeg2eog(new_eeg_feat, eog_feat)
        eog_fused_new_eeg = self.cross_eog2new_eeg(eog_feat, new_eeg_feat)

        eeg_fused_eog = self.cross_eeg2eog(eeg_feat, eog_feat)
        eog_fused_eeg = self.cross_eog2eeg(eog_feat, eeg_feat)

        
        
        eeg_state = self.state_proj_eeg(eeg_feat)
        new_eeg_state = self.state_proj_new_eeg(new_eeg_feat)
        eog_state = self.state_proj_eog(eog_feat)

        
        g_eeg_new = self.gate_eeg_new(eeg_state, new_eeg_state)
        g_eeg_eog = self.gate_eeg_eog(eeg_state, eog_state)

        g_new_eeg = self.gate_new_eeg(new_eeg_state, eeg_state)
        g_new_eog = self.gate_new_eog(new_eeg_state, eog_state)

        g_eog_eeg = self.gate_eog_eeg(eog_state, eeg_state)
        g_eog_new = self.gate_eog_new(eog_state, new_eeg_state)

        
        eeg_enhanced = eeg_feat \
                    + g_eeg_new * eeg_fused_new_eeg \
                    + g_eeg_eog * eeg_fused_eog

        new_eeg_enhanced = new_eeg_feat \
                        + g_new_eeg * new_eeg_fused_eeg \
                        + g_new_eog * new_eeg_fused_eog

        eog_enhanced = eog_feat \
                    + g_eog_eeg * eog_fused_eeg \
                    + g_eog_new * eog_fused_new_eeg

        combined = torch.cat([eeg_enhanced, new_eeg_enhanced, eog_enhanced], dim=-1)  
        projected = self.fusion_proj(combined)  

        lstm_out, _ = self.lstm(projected)
        attn_out = self.attn(lstm_out)
        
        attn_refined = self.attractor_rectifier(attn_out)
        pooled = attn_refined.mean(dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device, accumulate_steps=4):  
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    optimizer.zero_grad()

    for i, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        
        loss = criterion(logits, Y)
        loss = loss / accumulate_steps  
        loss.backward()

        if (i + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * X.size(0) * accumulate_steps  
        all_preds.append(preds)
        all_targets.append(Y)

        
        del X, Y, logits, probs, preds, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (i + 1) % accumulate_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / len(dataloader.dataset)
    
    
    f1 = F_metrics.f1_score(all_preds, all_targets, task="multiclass", num_classes=5, average='macro').item()
    acc = (all_preds == all_targets).float().mean().item()
    return avg_loss, acc, f1, all_preds, all_targets

def eval_epoch(model, dataloader, device):  
    model.eval()
    total_loss = 0
    all_preds_proba = []
    all_targets = []

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            
            f1 = F_metrics.f1_score(preds, Y, task="multiclass", num_classes=5, average='macro')
            loss = 1 - f1

            total_loss += loss.item() * X.size(0)
            all_preds_proba.append(probs.cpu().numpy())
            all_targets.append(Y.cpu().numpy())

            
            del X, Y, logits, probs, preds, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_preds_proba = np.concatenate(all_preds_proba, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(dataloader.dataset)

    num_classes = all_preds_proba.shape[1]
    best_thresholds = []
    for class_idx in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(all_targets == class_idx, all_preds_proba[:, class_idx])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[np.nanargmax(f1_scores)]
        best_thresholds.append(best_threshold)

    adjusted_proba = all_preds_proba - np.array(best_thresholds)
    all_preds = np.argmax(adjusted_proba, axis=1)

    all_preds_tensor = torch.tensor(all_preds, dtype=torch.long).to(device)
    all_targets_tensor = torch.tensor(all_targets, dtype=torch.long).to(device)
    
    
    f1 = F_metrics.f1_score(all_preds_tensor, all_targets_tensor, task="multiclass", num_classes=5, average='macro').item()

    acc = (all_preds == all_targets).mean()

    kappa = cohen_kappa_score(all_targets, all_preds)

    cm = confusion_matrix(all_targets, all_preds)

    num_classes = cm.shape[0]
    sensitivities = []
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)

    g_means = np.sqrt(np.array(sensitivities) * np.array(specificities))
    macro_g_mean = np.mean(g_means)

    
    return avg_loss, acc, f1, kappa, avg_sensitivity, avg_specificity, macro_g_mean, all_preds, all_targets

def load_data_from_npz_files(file_paths):
    all_x = []
    all_y = []
    file_paths.sort()
    for fpath in file_paths:
        
        data = np.load(fpath)
        
        if 'x_EEG(sec)' not in data or 'x_EOG(L)' not in data or 'x_EEG' not in data:
           
            continue
        eeg_data = data['x_EEG(sec)'].astype(np.float32)  
        
        if eeg_data.ndim == 3:
            eeg_data = eeg_data[:, :, 0]  
        eog_data = data['x_EOG(L)'].astype(np.float32)  
        new_eeg_data = data['x_EEG'].astype(np.float32)  
        
        if eeg_data.shape != eog_data.shape or eeg_data.shape != new_eeg_data.shape:
            
            continue
        X = np.stack([eeg_data, eog_data, new_eeg_data], axis=-1)
        
        
        Y = data['y']
        all_x.append(X)
        all_y.append(Y)
        

        if all_x:
            
            if len(all_x) % 20 == 0:
                gc.collect()
               

    if not all_x:
        
        return np.array([]), np.array([])

    
    batch_size = 10
    X_batches = []
    Y_batches = []
    for i in range(0, len(all_x), batch_size):
        X_batch = np.concatenate(all_x[i:i+batch_size], axis=0)
        Y_batch = np.concatenate(all_y[i:i+batch_size], axis=0)
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)
        del X_batch, Y_batch
        gc.collect()

    X = np.concatenate(X_batches, axis=0)
    Y = np.concatenate(Y_batches, axis=0)
    
    
    del all_x, all_y, X_batches, Y_batches
    gc.collect()
    
    return X, Y


def load_data_in_batches(file_paths, batch_size=10):
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Loading files", unit="batch"):
        batch_files = file_paths[i:i+batch_size]
        X_batch, Y_batch = load_data_from_npz_files(batch_files)
        yield X_batch, Y_batch
        del X_batch, Y_batch
        gc.collect()




if __name__ == "__main__":
    device_ids = args.device_ids
    device = torch.device(f"cuda:{device_ids[0]}")
    
    data_folder_train = args.data_folder_train
    data_folder_test = args.data_folder_test
   
    start_fold = args.start_fold

    npz_files = glob.glob(os.path.join(data_folder_test, "*.npz"))

    num_epoch = args.num_epoch

    
    random_seed = 52

   
    train_files, val_files = train_test_split(npz_files, test_size=0.3, random_state=random_seed)

    all_val_losses = []
    all_val_accuracies = []
    all_val_f1_scores = []
    all_val_kappas = []
    all_val_sensitivities = []
    all_val_specificities = []
    all_val_g_means = []    

    all_val_features = []
    all_val_preds = []
    all_val_targets = []

    
    print("train file list：")
    for train_file in train_files:
        print(train_file)
    print("test file list：")
    for val_file in val_files:
        print(val_file)
 
    X_val, Y_val = load_data_from_npz_files(val_files)
 
    val_dataset = EEGSleepDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    
    model = NIIASleepNet(d_model=128, num_classes=5, lstm_layers=2, dropout=0.5)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    
    criterion = HybridLoss(alpha=1, num_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5, last_epoch=-1)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []

    
    best_val_f1 = 0.0
    best_val_acc = 0.0  
    best_train_f1 = 0.0
    best_epoch = 0
    best_val_loss = 0.0
    best_train_acc = 0.0
    best_val_kappa = 0.0
    best_val_sensitivity = 0.0
    best_val_specificity = 0.0
    best_val_g_mean = 0.0

    
    for epoch in range(num_epoch):
        if epoch > 7:
            
            scheduler.step()

        total_train_loss = 0
        total_train_samples = 0
        all_train_preds = []
        all_train_targets = []

        
        for X_batch, Y_batch in load_data_in_batches(train_files, batch_size=5):
            
            batch_dataset = EEGSleepDataset(X_batch, Y_batch)
            batch_loader = DataLoader(batch_dataset, batch_size=16, shuffle=True, drop_last=True)

            
            train_loss, train_acc, train_f1, train_preds, train_targets = train_epoch(
                model, batch_loader, optimizer, criterion, device, accumulate_steps=4
            )

            total_train_loss += train_loss * len(batch_dataset)
            total_train_samples += len(batch_dataset)
            all_train_preds.append(train_preds)
            all_train_targets.append(train_targets)

            
            del batch_dataset, batch_loader, X_batch, Y_batch
            gc.collect()

        
        all_train_preds = torch.cat(all_train_preds)
        all_train_targets = torch.cat(all_train_targets)
        train_loss = total_train_loss / total_train_samples
        train_acc = (all_train_preds == all_train_targets).float().mean().item()
        train_f1 = F_metrics.f1_score(all_train_preds, all_train_targets, task="multiclass", num_classes=5, average='macro').item()

        
        val_loss, val_acc, val_f1, val_kappa, val_sensitivity, val_specificity, val_g_mean, val_preds, val_targets = eval_epoch(model, val_loader, device)

        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        print(f"Epoch [{epoch + 1}/{num_epoch}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
                f"Val Kappa: {val_kappa:.4f}, Val Sensitivity: {val_sensitivity:.4f}, "
                f"Val Specificity: {val_specificity:.4f}, Val G-mean: {val_g_mean:.4f}")    

        
        combined_score = val_acc * 1 + val_f1 * 0
        best_combined_score = best_val_acc * 1 + best_val_f1 * 0
        
        if combined_score > best_combined_score:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_train_f1 = train_f1
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_train_acc = train_acc
            best_val_kappa = val_kappa
            best_val_sensitivity = val_sensitivity
            best_val_specificity = val_specificity
            best_val_g_mean = val_g_mean

            
            fold_dir = args.fold_dir
            os.makedirs(fold_dir, exist_ok=True)

            
            model_filename = f"best_model_epoch_{best_epoch}_train_acc_{best_train_acc:.4f}_val_acc_{best_val_acc:.4f}_train_f1_{best_train_f1:.4f}_val_f1_{best_val_f1:.4f}.pth"
            model_path = os.path.join(fold_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved. Model epoch: {best_epoch}, Train Acc: {best_train_acc:.4f}, Val Acc: {best_val_acc:.4f}, Train F1: {best_train_f1:.4f}, Val F1: {best_val_f1:.4f}")   

           
            
            
            train_class_f1_scores = f1_score(train_targets.cpu().numpy(), train_preds.cpu().numpy(), average=None)
            class_names = ["Wake", "N1", "N2", "N3", "REM"]
            print("train F1 :")
            for cls_name, f1 in zip(class_names, train_class_f1_scores):
                print(f"{cls_name}: {f1:.4f}")
            
            
            class_f1_scores = f1_score(val_targets, val_preds, average=None)
            class_names = ["Wake", "N1", "N2", "N3", "REM"]
            print("test F1 :")
            for cls_name, f1 in zip(class_names, class_f1_scores):
                print(f"{cls_name}: {f1:.4f}")      

            
            cm = confusion_matrix(val_targets, val_preds)
            
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Wake", "N1", "N2", "N3", "REM"])
            fig, ax = plt.subplots(figsize=(8, 6))
            
            disp.plot(ax=ax, cmap="Blues", values_format='.2f')
            
            num_classes = len(disp.display_labels)
            ax.set_xticks(np.arange(num_classes))
            ax.set_xticklabels(disp.display_labels, rotation=45, ha='right')
            
            ax.set_yticks(np.arange(num_classes))
            ax.set_yticklabels(disp.display_labels)
            
            ax.set_xlim(-0.5, num_classes - 0.5)
            ax.set_ylim(num_classes - 0.5, -0.5)
            
            ax.grid(False)
            plt.title("Confusion Matrix")
            plt.tight_layout()  
            plt.savefig(os.path.join(fold_dir, f"confusion_matrix_CNNLFW_single_split.png"))
            plt.close()
            
            
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            class_names = ["Wake", "N1", "N2", "N3", "REM"]
            print("acc:")
            for cls_name, acc in zip(class_names, class_accuracies):
                print(f"{cls_name}: {acc:.4f}")

            
            report = classification_report(val_targets, val_preds, target_names=["Wake", "N1", "N2", "N3", "REM"])
            print(report)

            
            all_val_features.clear()
            all_val_preds.clear()
            all_val_targets.clear()

            
            with torch.no_grad():
                model.eval()
                fold_val_features = []
                for X, _ in val_loader:
                    X = X.to(device)
                    
                    eeg = X[:, :, 0].unsqueeze(1)
                    eog = X[:, :, 1].unsqueeze(1)
                    new_eeg = X[:, :, 2].unsqueeze(1)

                    
                    eeg_feat   = model.module.eeg_cnn(eeg).permute(0, 2, 1)
                    new_eeg_feat = model.module.new_eeg_cnn(new_eeg).permute(0, 2, 1)
                    target_len = eeg_feat.size(1)
                    eog_feat   = model.module.eog_cnn(eog, target_len=target_len)

                    
                    
                    eeg_state   = model.module.state_proj_eeg(eeg_feat)
                    new_eeg_state = model.module.state_proj_new_eeg(new_eeg_feat)
                    eog_state   = model.module.state_proj_eog(eog_feat)

                    g_eeg_new   = model.module.gate_eeg_new(eeg_state, new_eeg_state)
                    g_eeg_eog   = model.module.gate_eog_eeg(eeg_state, eog_state)
                    g_new_eeg   = model.module.gate_new_eeg(new_eeg_state, eeg_state)
                    g_new_eog   = model.module.gate_new_eog(new_eeg_state, eog_state)
                    g_eog_eeg   = model.module.gate_eog_eeg(eog_state, eeg_state)
                    g_eog_new   = model.module.gate_eog_new(eog_state, new_eeg_state)

                    eeg_enhanced = eeg_feat + g_eeg_new * model.module.cross_eeg2new_eeg(eeg_feat, new_eeg_feat) \
                                                 + g_eeg_eog * model.module.cross_eeg2eog(eeg_feat, eog_feat)
                    new_eeg_enhanced = new_eeg_feat + g_new_eeg * model.module.cross_new_eeg2eeg(new_eeg_feat, eeg_feat) \
                                                       + g_new_eog * model.module.cross_new_eeg2eog(new_eeg_feat, eog_feat)
                    eog_enhanced = eog_feat + g_eog_eeg * model.module.cross_eog2eeg(eog_feat, eeg_feat) \
                                            + g_eog_new * model.module.cross_eog2new_eeg(eog_feat, new_eeg_feat)

                    combined   = torch.cat([eeg_enhanced, new_eeg_enhanced, eog_enhanced], dim=-1)
                    projected  = model.module.fusion_proj(combined)
                    lstm_out, _= model.module.lstm(projected)
                    attn_out   = model.module.attn(lstm_out)
                    
                    attn_refined = model.module.attractor_rectifier(attn_out)

                    
                    pooled = attn_refined.mean(dim=1)          
                    fold_val_features.append(pooled.cpu().numpy())

                fold_val_features = np.concatenate(fold_val_features, axis=0)
                all_val_features.append(fold_val_features)
                all_val_preds.append(val_preds)
                all_val_targets.append(val_targets)

        
        del all_train_preds, all_train_targets
        gc.collect()
    
    
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(fold_dir, f"loss_plot_CNNLFW_single_split.png"))
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(fold_dir, f"accuracy_plot_CNNLFW_single_split.png"))
    plt.close()

    plt.figure()
    plt.plot(train_f1_scores, label="Train F1")
    plt.plot(val_f1_scores, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(os.path.join(fold_dir, f"f1_score_plot_CNNLFW_single_split.png"))
    plt.close()

    
    print(f"best performance:")
    print(f"val_loss: {best_val_loss:.4f}")
    print(f"val_acc: {best_val_acc:.4f}")
    print(f"val_f1: {best_val_f1:.4f}")
    print(f"Cohen's Kappa: {best_val_kappa:.4f}")
    print(f"Sensitivity: {best_val_sensitivity:.4f}")
    print(f"Specificity: {best_val_specificity:.4f}")
    print(f"Macro-averaged G-mean: {best_val_g_mean:.4f}")

    
    all_val_losses.append(best_val_loss)
    all_val_accuracies.append(best_val_acc)
    all_val_f1_scores.append(best_val_f1)
    all_val_kappas.append(best_val_kappa)
    all_val_sensitivities.append(best_val_sensitivity)
    all_val_specificities.append(best_val_specificity)
    all_val_g_means.append(best_val_g_mean) 




    
    all_val_features = np.concatenate(all_val_features, axis=0)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_targets = np.concatenate(all_val_targets, axis=0)

    
    sample_ratio = 0.07  
    sample_indices = np.random.choice(len(all_val_features), int(len(all_val_features) * sample_ratio), replace=False)
    sampled_val_features = all_val_features[sample_indices]
    sampled_val_preds = all_val_preds[sample_indices]
    sampled_val_targets = all_val_targets[sample_indices]

    
    tsne_2d = TSNE(n_components=2, random_state=42)
    reduced_features_2d = tsne_2d.fit_transform(sampled_val_features)

    
    def add_jitter(data, jitter_amount=0.5):
        return data + np.random.normal(0, jitter_amount, data.shape)

    
    reduced_features_2d_jittered = add_jitter(reduced_features_2d)


    
    class_names = ["Wake", "N1", "N2", "N3", "REM"]
    base_cmap = plt.colormaps['viridis']
    colors = ListedColormap(base_cmap(np.linspace(0, 1, len(class_names))))

    plt.figure(figsize=(15, 6))

    
    plt.subplot(1, 2, 1)
    for i in range(len(class_names)):
        indices = sampled_val_targets == i
        plt.scatter(reduced_features_2d[indices, 0], reduced_features_2d[indices, 1], 
                    color=colors(i), label=class_names[i], s=10, alpha=0.6)
    plt.title('2D True Labels')
    plt.legend()

    
    plt.subplot(1, 2, 2)
    for i in range(len(class_names)):
        indices = sampled_val_preds == i
        plt.scatter(reduced_features_2d[indices, 0], reduced_features_2d[indices, 1], 
                    color=colors(i), label=class_names[i], s=10, alpha=0.6)
    plt.title('2D Predicted Labels')
    plt.legend()

    plt.tight_layout()
    plot_2d_path = os.path.join(log_dir, 'classification_scatter_plot_2d_sampled.png')
    plt.savefig(plot_2d_path)
    plt.close()

   

    
    tsne_3d = TSNE(n_components=3, random_state=42)
    reduced_features_3d = tsne_3d.fit_transform(sampled_val_features)

    class_names = ["Wake", "N1", "N2", "N3", "REM"]

    
    sampled_val_targets_names = [class_names[i] for i in sampled_val_targets]
    sampled_val_preds_names = [class_names[i] for i in sampled_val_preds]

    
    color_sequence = px.colors.qualitative.Plotly[:len(class_names)]


    
    fig = px.scatter_3d(
        x=reduced_features_3d[:, 0],
        y=reduced_features_3d[:, 1],
        z=reduced_features_3d[:, 2],
        color=sampled_val_targets_names,
        color_discrete_sequence=color_sequence,
        labels={'color': 'True Labels'},
        title="3D True Labels",
        category_orders={'color': class_names},
        size_max=8,  
        size=[6] * len(sampled_val_targets),
        opacity=0.9
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    plot_3d_true_path = os.path.join(log_dir, 'classification_scatter_plot_3d_true_sampled.html')
    fig.write_html(plot_3d_true_path)

    
    fig = px.scatter_3d(
        x=reduced_features_3d[:, 0],
        y=reduced_features_3d[:, 1],
        z=reduced_features_3d[:, 2],
        color=sampled_val_preds_names,
        color_discrete_sequence=color_sequence,
        labels={'color': 'Predicted Labels'},
        title="3D Predicted Labels",
        category_orders={'color': class_names},
        size_max=8,  
        size=[6] * len(sampled_val_preds),
        opacity=0.9
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    plot_3d_pred_path = os.path.join(log_dir, 'classification_scatter_plot_3d_pred_sampled.html')
    fig.write_html(plot_3d_pred_path)

    
   

    
    tsne_2d = TSNE(n_components=2, random_state=42)
    reduced_features_2d = tsne_2d.fit_transform(all_val_features)

    
    class_names = ["Wake", "N1", "N2", "N3", "REM"]
    base_cmap = plt.colormaps['viridis']
    colors = ListedColormap(base_cmap(np.linspace(0, 1, len(class_names))))

    plt.figure(figsize=(15, 6))

    
    plt.subplot(1, 2, 1)
    for i in range(len(class_names)):
        indices = all_val_targets == i
        plt.scatter(reduced_features_2d[indices, 0], reduced_features_2d[indices, 1], 
                    color=colors(i), label=class_names[i], s=10, alpha=0.6)
    plt.title('2D True Labels')
    plt.legend()

    
    plt.subplot(1, 2, 2)
    for i in range(len(class_names)):
        indices = all_val_preds == i
        plt.scatter(reduced_features_2d[indices, 0], reduced_features_2d[indices, 1], 
                    color=colors(i), label=class_names[i], s=10, alpha=0.6)
    plt.title('2D Predicted Labels')
    plt.legend()

    plt.tight_layout()
    plot_2d_path = os.path.join(log_dir, 'classification_scatter_plot_2d_all.png')
    plt.savefig(plot_2d_path)
    plt.close()

   

    
    tsne_3d = TSNE(n_components=3, random_state=42)
    reduced_features_3d = tsne_3d.fit_transform(all_val_features)


    class_names = ["Wake", "N1", "N2", "N3", "REM"]

    
    all_val_targets_names = [class_names[i] for i in all_val_targets]
    all_val_preds_names = [class_names[i] for i in all_val_preds]

    
    color_sequence = px.colors.qualitative.Plotly[:len(class_names)]


    
    fig = px.scatter_3d(
        x=reduced_features_3d[:, 0],
        y=reduced_features_3d[:, 1],
        z=reduced_features_3d[:, 2],
        color=all_val_targets_names,
        color_discrete_sequence=color_sequence,
        labels={'color': 'True Labels'},
        title="3D True Labels",
        category_orders={'color': class_names},
        size_max=8,  
        size=[6] * len(all_val_targets),
        opacity=0.9
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    plot_3d_true_path = os.path.join(log_dir, 'classification_scatter_plot_3d_true_all.html')
    fig.write_html(plot_3d_true_path)

    
    fig = px.scatter_3d(
        x=reduced_features_3d[:, 0],
        y=reduced_features_3d[:, 1],
        z=reduced_features_3d[:, 2],
        color=all_val_preds_names,
        color_discrete_sequence=color_sequence,
        labels={'color': 'Predicted Labels'},
        title="3D Predicted Labels",
        category_orders={'color': class_names},
        size_max=8,  
        size=[6] * len(all_val_preds),
        opacity=0.9
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    plot_3d_pred_path = os.path.join(log_dir, 'classification_scatter_plot_3d_pred_all.html')
    fig.write_html(plot_3d_pred_path)

   



    
    avg_val_loss = np.mean(all_val_losses)
    avg_val_acc = np.mean(all_val_accuracies)
    avg_val_f1 = np.mean(all_val_f1_scores)
    avg_val_kappa = np.mean(all_val_kappas)
    avg_val_sensitivity = np.mean(all_val_sensitivities)
    avg_val_specificity = np.mean(all_val_specificities)
    avg_val_g_mean = np.mean(all_val_g_means)    
    print(f"\nSingle Split - Average Val Loss: {avg_val_loss:.4f}, Average Val Acc: {avg_val_acc:.4f}, Average Val F1: {avg_val_f1:.4f}, "
          f"Average Val Kappa: {avg_val_kappa:.4f}, Average Val Sensitivity: {avg_val_sensitivity:.4f}, "
          f"Average Val Specificity: {avg_val_specificity:.4f}, Average Val G-mean: {avg_val_g_mean:.4f}")
    
    
    

    
    log_file.close()
