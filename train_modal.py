import modal 
from torch import nn
import torch 
import sys

sys.path.append("/data")

img = modal.Image.debian_slim().pip_install("torch", "torchaudio", "torchvision" , "librosa", "numpy", "pandas", "matplotlib")
vol = modal.Volume.from_name("data",create_if_missing=True)

app = modal.App(image=img, volumes={"/data": vol})


@app.function(volumes={"/data":vol})
def download_and_extract_zip():
    import requests
    import zipfile
    import os 
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_file = os.path.join("/data", "data.zip")

    if os.path.exists("/data/ESC-50-master"):
        print("Data already exists. Skipping download.")
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(zip_file, 'wb') as f:
            f.write(response.content)
            print("successfully downloaded data.zip")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("/data")
            print("successfully extracted data.zip")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")
    
    vol.commit()
    
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(AudioClassifier, self).__init__() 
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1 ,32 , kernel_size=(3,3) , stride=(1,1),padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

        
        
    def forward(self,x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        with torch.no_grad():
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)
    


@app.function(volumes={"/data":vol},
              timeout=86400,
              gpu="T4",
              cpu=4,
              memory=8192
              )
def train():
    import os 
    import pandas as pd
    import numpy as np
    import random
    import torchvision
    import torchaudio
    from dataset import ESC50Dataset
    from torch.utils.data import DataLoader
    from torch import nn
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    data_dir = os.path.join("/data","ESC-50-master")
    audio_dir = os.path.join(data_dir, "audio")
    table_dir = os.path.join(data_dir, "meta")

    df = pd.read_csv(os.path.join(table_dir, "esc50.csv"))
    df.drop(columns=["src_file","take","esc10"], inplace=True)

    folds = df["fold"].unique()
    val_fold = np.random.choice(folds)
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((128,128)), #resize should never be applied to audio data as it can distort the frequency information, 
        #the incoming mel spectrogram is already 128 by 128 (made sure by choosing hop_length=625) but the best thing to do instead of resizing is center cropping
        torchvision.transforms.CenterCrop((128,128)),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=8),
        torchaudio.transforms.TimeMasking(time_mask_param=12)
    ])

    train_dataset = ESC50Dataset(train_df, audio_dir,transform=transform)
    val_dataset = ESC50Dataset(val_df, audio_dir,transform=None)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
   
    model = AudioClassifier(num_classes=50)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=4e-4)
    best_val_loss = float('inf')
    load_weight = True #sometimes important to set to false after making big changes
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    if os.path.exists(os.path.join("/data", "best_model.pth")) and load_weight:
        print("Loading best model weights...")
        best_chkpt = torch.load(os.path.join("/data", "best_model.pth"), map_location=device)
        model.load_state_dict(best_chkpt["model_state_dict"])
        optimizer.load_state_dict(best_chkpt["optimizer_state_dict"])
        best_val_loss = best_chkpt["val_loss"]
        if "scheduler_state_dict" in best_chkpt:
            scheduler.load_state_dict(best_chkpt["scheduler_state_dict"])

        print(f"Loaded best weight with loss: {best_val_loss}")
   

    
    
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Training Started...")
    print(f"Total number of trainable parameters: {p_count:,}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch , y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        val_loss = validate(model, val_loader, criterion, device)
        chkpt = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "scheduler_state_dict": scheduler.state_dict()}
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Best Validation found at epoch {epoch+1} with loss: {best_val_loss:.4f}. Saving model...")  
            
            
            torch.save(chkpt, os.path.join("/data", "best_model.pth"))
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
            torch.save(chkpt, os.path.join("/data", f"latest_model.pth"))

        vol.commit()  
        scheduler.step()

    print("Training Completed.")
@app.local_entrypoint()
def main():
    download_and_extract_zip.spawn()
    train.spawn()


