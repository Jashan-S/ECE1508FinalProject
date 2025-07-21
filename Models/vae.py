import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class MelDataset(Dataset):
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob("*.pt"))
        self.speakers = sorted(list({f.stem.split('_')[0] for f in self.files}))
        self.spk2idx = {spk: i for i, spk in enumerate(self.speakers)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        mel = sample['mel']
        speaker = sample['speaker_id']
        speaker_idx = self.spk2idx[speaker]
        return mel, speaker_idx

def collate_fn(batch, target_len=400):
    mels, spk_ids = zip(*batch)
    padded = []

    for mel in mels:
        if mel.shape[1] >= target_len:
            mel_fixed = mel[:, :target_len]
        else:
            pad_width = target_len - mel.shape[1]
            mel_fixed = F.pad(mel, (0, pad_width))
        padded.append(mel_fixed)

    mels_tensor = torch.stack(padded)
    spk_ids_tensor = torch.tensor(spk_ids)
    return mels_tensor, spk_ids_tensor

# --- VAE Model ---
class VAE(nn.Module):
    def __init__(self, input_dim=80, latent_dim=128, speaker_emb_dim=32, num_speakers=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.to_mu = nn.Linear(128, latent_dim)
        self.to_logvar = nn.Linear(128, latent_dim)

        self.speaker_embed = nn.Embedding(num_speakers, speaker_emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + speaker_emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim * 400),
            nn.Tanh()
        )
        self.input_dim = input_dim

    def forward(self, x, speaker_id):
        h = self.encoder(x).squeeze(-1)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        speaker_emb = self.speaker_embed(speaker_id)
        z_cat = torch.cat([z, speaker_emb], dim=-1)
        out = self.decoder(z_cat).view(-1, self.input_dim, 400)
        return out, mu, logvar

# --- Loss ---
def vae_loss(recon, x, mu, logvar, beta=0.01):
    recon_loss = F.mse_loss(recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()

# --- Training ---
def train(data_dir="../data/processed",
          latent_dim=256,
          speaker_emb_dim=64,
          batch_size=16,
          epochs=20,
          learning_rate=1e-3,
          beta=0.01,
          save_dir="../checkpoints"):

    dataset = MelDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = VAE(input_dim=80, latent_dim=latent_dim, speaker_emb_dim=speaker_emb_dim, num_speakers=len(dataset.speakers)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        for x, spk in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, spk = x.to(DEVICE), spk.to(DEVICE)
            recon, mu, logvar = model(x, spk)
            loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_l
            total_kl += kl_l

        print(f"Epoch {epoch+1}: Total={total_loss:.2f} | Recon={total_recon:.2f} | KL={total_kl:.2f}")

    model_name = f"vae_lat{latent_dim}_spk{speaker_emb_dim}_ep{epochs}_beta{int(beta*1000):03}.pt"
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")
