

import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


import matplotlib as mpl



current_font_size = mpl.rcParams['font.size']
mpl.rcParams.update({'font.size': current_font_size * 1.25})



def plot_confusion_matrix(true_labels, pred_labels, output_dir, filename, log_file_path):
    cm = confusion_matrix(true_labels, pred_labels)
    class_names = ["Wake", "N1", "N2", "N3", "REM"]

    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
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
    cm_plot_path = os.path.join(output_dir, f"confusion_matrix_{filename}.png")
    plt.savefig(cm_plot_path)
    plt.close()


def print_and_log(message, log_file_path):
    print(message)
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    if total_params < 1e3:
        return f"{total_params} params"
    elif total_params < 1e6:
        return f"{total_params / 1e3:.2f} K params"
    elif total_params < 1e9:
        return f"{total_params / 1e6:.2f} M params"
    else:
        return f"{total_params / 1e9:.2f} B params"

def predict(model, dataloader, device):
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(Y.cpu().numpy())
            
            
            del X, Y, logits, probs, preds
            torch.cuda.empty_cache()
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return all_preds, all_targets




def calculate_metrics(true_labels, pred_labels):
    
    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True)
    
    
    unique_classes = np.unique(true_labels)
    valid_classes = []
    class_metrics = {}
    for class_label in range(5):  
        if class_label in unique_classes:
            valid_classes.append(class_label)
            class_metrics[class_label] = {
                'acc': report[str(class_label)]['precision'],
                'f1': report[str(class_label)]['f1-score']
            }
        else:
            class_metrics[class_label] = {
                'acc': None,
                'f1': None
            }
    
    
    if valid_classes:
        avg_f1 = np.mean([class_metrics[cls]['f1'] for cls in valid_classes])
    else:
        avg_f1 = 0
    
    return acc, avg_f1, class_metrics




class EEGSleepDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)  
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]




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



class CNNSleepLSTMModel(nn.Module):
    def __init__(self, d_model=128, num_classes=5, lstm_layers=2, dropout=0.5):
        super(CNNSleepLSTMModel, self).__init__()

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




def load_single_npz_file(file_path):
    
    data = np.load(file_path)
    
    
    required_keys = ['x_EEG(sec)', 'x_EOG(L)', 'x_EEG', 'y']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"file {file_path} lost: {key}")
    
    
    eeg_data = data['x_EEG(sec)'].astype(np.float32)
    eog_data = data['x_EOG(L)'].astype(np.float32)
    new_eeg_data = data['x_EEG'].astype(np.float32)
    labels = data['y']
    
    
    if eeg_data.shape != eog_data.shape or eeg_data.shape != new_eeg_data.shape:
        raise ValueError(f"file {file_path} : Inconsistent"
                         f"EEG: {eeg_data.shape}, EOG: {eog_data.shape}, New EEG: {new_eeg_data.shape}")
    
    
    X = np.stack([eeg_data, eog_data, new_eeg_data], axis=-1)
    
    return X, labels


def predict(model, dataloader, device):
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(Y.cpu().numpy())
            
            
            del X, Y, logits, probs, preds
            torch.cuda.empty_cache()
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return all_preds, all_targets




def plot_sleep_stages(true_labels, pred_labels, output_dir, filename):
    
    
    plt.figure(figsize=(15, 6))
    plt.plot(true_labels, color="
    
    plt.title(f'True hypnogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Sleep Stage')
    plt.yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, len(true_labels))
    plt.legend(['True Labels'])
    plt.tight_layout()
    true_plot_path = os.path.join(output_dir, f'true_labels_{filename}.png')
    plt.savefig(true_plot_path, dpi=300)
    plt.close()

    
    plt.figure(figsize=(15, 6))
    plt.plot(pred_labels, color="
    
    plt.title(f'Predicted hypnogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Sleep Stage')
    plt.yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, len(pred_labels))
    plt.legend(['Predicted Labels'])
    plt.tight_layout()
    pred_plot_path = os.path.join(output_dir, f'predicted_labels_{filename}.png')
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()

    
    plt.figure(figsize=(15, 6))
    plt.plot(true_labels, color="
    
    
    plt.plot(pred_labels + 0.1, color="
    
    
    plt.title(f'True and predicted hypnograms')
    
    plt.xlabel('Sample Index')
    
    plt.ylabel('Sleep Stage')
    
    plt.yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    
    plt.ylim(-0.2, 4.3)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, len(true_labels))
    plt.legend()
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, f'combined_labels_{filename}.png')
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()

    print(f"Plots saved to: {true_plot_path}, {pred_plot_path} and {combined_plot_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='ISATSleepNet_test')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file (required)')
    parser.add_argument('--npz_file_path', type=str, required=True, help='Path to the NPZ data file (required)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output results (required)')
    parser.add_argument('--device', type=str, default=None, help='Running device (e.g. cuda:0/cpu), auto-detected by default')
    args = parser.parse_args()
    
    model_path = args.model_path
    npz_file_path = args.npz_file_path
    output_dir = args.output_dir
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    log_file_path = os.path.join(output_dir, 'test_log.txt')
    
    
    npz_file_name = os.path.basename(npz_file_path)
    
    
    model = CNNSleepLSTMModel(d_model=128, num_classes=5, lstm_layers=2, dropout=0.5)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
  
    param_count = count_model_parameters(model)
    
    X, Y = load_single_npz_file(npz_file_path)
    print_and_log(f" X={X.shape}, Y={Y.shape}", log_file_path)
    
    
    dataset = EEGSleepDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    
    print_and_log("pridicting...", log_file_path)
    pred_labels, true_labels = predict(model, dataloader, device)
    print_and_log(f" {pred_labels.shape}", log_file_path)
    
    
    acc, avg_f1, class_metrics = calculate_metrics(true_labels, pred_labels)
    print_and_log(f"Accuracy: {acc:.4f}", log_file_path)
    print_and_log(f" MF1  (Macro F1): {avg_f1:.4f}", log_file_path)
    
   
    
    
    filename = os.path.basename(npz_file_path).replace('.npz', '')
    plot_confusion_matrix(true_labels, pred_labels, output_dir, filename, log_file_path)
    
    
    
    filename = os.path.basename(npz_file_path).replace('.npz', '')
    plot_sleep_stages(true_labels, pred_labels, output_dir, filename)
    
