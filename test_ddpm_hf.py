import torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet1DModel
import numpy as np
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from dataclasses import dataclass
import torch.nn.functional as F
from positional_embeddings_cuda import PositionalEmbedding

# training config
@dataclass
class TrainingConfig:
    num_points = 2
    train_batch_size = 64
    eval_batch_size = 1000
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_model_epochs = 10
    mixed_precision = 'fp16'
    output_dir = 'ddpm-circle-points'
    seed = 0

config = TrainingConfig()

# our circular points dataloader
class LinePointDataset(Dataset):
    def __init__(self, num_points=2, radius=1, noise=0.05):
        self.num_points = num_points
        self.radius = radius
        self.noise = noise
        self.data = self.generate_data()

    def generate_data(self):
        l = [[1,0], [-1,0]]
        images = torch.tensor(np.stack(l, axis=1), dtype=torch.float32)
        return images

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.data[idx]
    
dataset = LinePointDataset(num_points=config.num_points)
data = dataset.data
dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# our model to run diffusion
class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

model = MLP(
        hidden_size=128,
        hidden_layers=3,
        emb_size=128,
        time_emb="sinusoidal",
        input_emb="sinusoidal")

# initialize noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# initialize our optimizer
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_points = batch
            noise = torch.randn(clean_points.shape).to(clean_points.device)
            bs = clean_points.shape[0]

            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_points.device).long()

            noisy_points = noise_scheduler.add_noise(clean_points, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_points, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save(model.state_dict(), f"{config.output_dir}/model_epoch_{epoch}.pt")

    return model

trained_model = train_loop(config, model, noise_scheduler, optimizer, dataloader, lr_scheduler)

@torch.no_grad()
def sample_points(model, num_points=1000):
    model.eval()
    device = next(model.parameters()).device
    
    # Start from Gaussian noise
    x = torch.randn(num_points, 2).to(device)

    for t in reversed(range(noise_scheduler.num_train_timesteps)):
        timesteps = torch.full((num_points,), t, dtype=torch.long, device=device)
        noise_pred = model(x, timesteps)
        x = noise_scheduler.step(noise_pred.squeeze(0), t, x).prev_sample

    return x.cpu().numpy()

# Generate and plot samples
samples = sample_points(trained_model, num_points=20)
plt.figure(figsize=(10, 10))
plt.scatter(data[:, 0], data[:, 1], c='r', label='Original data')
plt.scatter(samples[:, 0], samples[:, 1], c='b', alpha=0.1, label='Generated samples')
plt.legend()
plt.title("Original Data vs Generated Samples")
plt.axis('equal')
plt.show()