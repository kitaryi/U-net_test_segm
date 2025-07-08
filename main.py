import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout(0.1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  #Входной канал 256 (128+128 из skip connection)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout5 = nn.Dropout(0.1)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  #Входной канал 128 (64+64 из skip connection)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout6 = nn.Dropout(0.1)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  #Входной канал 64 (32+32 из skip connection)
        self.conv14 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout7 = nn.Dropout(0.1)

        self.conv15 = nn.Conv2d(32, 1, kernel_size=1)

        #Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv2 = self.dropout1(conv2)
        pool1 = self.pool1(conv2)  #shape: [batch_size, 32, H/2, W/2]

        conv3 = F.relu(self.conv3(pool1))
        conv4 = F.relu(self.conv4(conv3))
        conv4 = self.dropout2(conv4)
        pool2 = self.pool2(conv4)  #shape: [batch_size, 64, H/4, W/4]

        conv5 = F.relu(self.conv5(pool2))
        conv6 = F.relu(self.conv6(conv5))
        conv6 = self.dropout3(conv6)
        pool3 = self.pool3(conv6)  #shape: [batch_size, 128, H/8, W/8]

        conv7 = F.relu(self.conv7(pool3))
        conv8 = F.relu(self.conv8(conv7))
        conv8 = self.dropout4(conv8)  #shape: [batch_size, 256, H/8, W/8]

        # Decoder
        up1 = self.up1(conv8)  #shape: [batch_size, 128, H/4, W/4]
        #Skip connection: concatenate up1 and conv6
        up1 = torch.cat([up1, conv6], dim=1)  #shape: [batch_size, 256, H/4, W/4]
        conv9 = F.relu(self.conv9(up1))
        conv10 = F.relu(self.conv10(conv9))
        conv10 = self.dropout5(conv10)  #shape: [batch_size, 128, H/4, W/4]

        up2 = self.up2(conv10)  #shape: [batch_size, 64, H/2, W/2]
        #Skip connection: concatenate up2 and conv4
        up2 = torch.cat([up2, conv4], dim=1)  #shape: [batch_size, 128, H/2, W/2]
        conv11 = F.relu(self.conv11(up2))
        conv12 = F.relu(self.conv12(conv11))
        conv12 = self.dropout6(conv12)  #shape: [batch_size, 64, H/2, W/2]

        up3 = self.up3(conv12)  #shape: [batch_size, 32, H, W]
        #Skip connection: concatenate up3 and conv2
        up3 = torch.cat([up3, conv2], dim=1)  #shape: [batch_size, 64, H, W]
        conv13 = F.relu(self.conv13(up3))
        conv14 = F.relu(self.conv14(conv13))
        conv14 = self.dropout7(conv14)  #shape: [batch_size, 32, H, W]

        conv15 = self.conv15(conv14)  #shape: [batch_size, 1, H, W]
        outputs = conv15 # shape: [batch_size, 1, H, W]
        return outputs


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, expected_size=(1216, 1824)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.expected_size = expected_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = self.image_files[idx]
            base_name = img_name.split(".JPG")[0]
            mask_name = base_name + ".png"
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, mask_name)

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            #print(f"Маска после загрузки: {np.unique(mask)}")

            if image is None:
                print(f"Не удалось загрузить изображение: {img_path}")
                return None  # Возвращаем None

            if mask is None:
                print(f"Не удалось загрузить маску: {mask_path}")
                return None  # Возвращаем None

            image = Image.fromarray(image)
            mask = Image.fromarray(mask)

            original_width = 1216
            original_height = 1824
            width, height = image.size

            if width > height:
                image = image.rotate(90, expand=True)
                mask = mask.rotate(90, expand=True)
            width, height = image.size

            if (width, height) != (original_width, original_height):
                image = image.resize((original_width, original_height), Image.Resampling.LANCZOS)
                mask = mask.resize((original_width, original_height), Image.Resampling.NEAREST)

            #Преобразуем Image в numpy array
            image = np.array(image)
            mask = np.array(mask)

            #Нормализуем маску
            #print(f"Маска после нормализации и округления: {np.unique(mask)}")
            mask = np.expand_dims(mask, axis=-1)

            if self.transform:
                image = self.transform(image)

            mask = torch.from_numpy(mask).float().permute(2, 0, 1)
            #print(f"Маска после преобразования в тензор: {torch.unique(mask)}")
            return image, mask

        except Exception as e:
            print(f"Ошибка при обработке {img_name}: {e}")
            return None #Возвращаем None


def collate_fn(batch):

    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.zeros((1, 3, 256, 256)), torch.zeros((1, 1, 256, 256))
    images, masks = zip(*batch)
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)

    return images, masks


transform = transforms.Compose([
    transforms.ToTensor(),
])




train_image_dir = os.path.join('C:/Users/Developer/PycharmProjects/Segm_test/train/images')
train_mask_dir = os.path.join('C:/Users/Developer/PycharmProjects/Segm_test/train/masks')
val_image_dir = os.path.join('C:/Users/Developer/PycharmProjects/Segm_test/val/images')
val_mask_dir = os.path.join('C:/Users/Developer/PycharmProjects/Segm_test/val/masks')


train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)

val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,collate_fn=collate_fn)

print(torch.cuda.get_device_name(0))
device = torch.device("cuda")
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

num_epochs = 10

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(val_loader.dataset)

best_val_loss = float('inf')
accumulation_steps = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        #print(f"Unique mask values: {torch.unique(masks)}")
        optimizer.zero_grad()

        outputs = model(images)
        #print(f"Min output value: {outputs.min()}")
        #print(f"Max output value: {outputs.max()}")
        loss = criterion(outputs, masks)
        loss = loss / accumulation_steps
        loss.backward()
        running_loss += loss.item()*accumulation_steps

        if (i + 1) % accumulation_steps == 0:
            torch.cuda.empty_cache()
            optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

    # Оценка на проверочном наборе данных в конце каждой эпохи
    val_loss = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_unet.pth')



print('Конец, баста!')


#Загружаем лучшие веса в модель
model.load_state_dict(torch.load('best_unet.pth'))
torch.save(model.state_dict(), 'unet.pth')