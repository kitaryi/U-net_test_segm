import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


# Определение примерной структуры unet
# https://drive.google.com/file/d/1bG_0SyGJ8EYRZv1Ch4bymEptv8zDqlYk/view?usp=sharing
# https://drive.google.com/file/d/1R5hZ7Uwm4VywNRcw-1pp0VbkCr7OyOOW/view?usp=sharing

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # "Encoder"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # "Bottleneck"
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # "Decoder"
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv15 = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        x = self.pool1(x1)
        x2 = torch.relu(self.conv4(torch.relu(self.conv3(x))))
        x = self.pool2(x2)
        x3 = torch.relu(self.conv6(torch.relu(self.conv5(x))))
        x = self.pool3(x3)
        x = torch.relu(self.conv8(torch.relu(self.conv7(x))))
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = torch.relu(self.conv10(torch.relu(self.conv9(x))))
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = torch.relu(self.conv12(torch.relu(self.conv11(x))))
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = torch.relu(self.conv14(torch.relu(self.conv13(x))))
        x = self.conv15(x)
        x = self.sigmoid(x)
        return x

# Преобразование изображений
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        if "_bright" in img_name:
            base_name = img_name.split("_bright")[0]
            mask_name = base_name + "_bright.png"
        elif "_contrast" in img_name:
            base_name = img_name.split("_contrast")[0]
            mask_name = base_name + "_contrast.png"
        else:
            base_name = img_name.split(".JPG")[0]
            mask_name = base_name + ".png"
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


transform = transforms.Compose([
    transforms.ToTensor(),
])


im_dir = 'D:/your_dataset'
mask_dir = 'D:/your_dataset_result'
train_dataset = SegmentationDataset(im_dir, mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
print(torch.cuda.get_device_name(0))
device = torch.device("cuda")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, masks, outputs, loss
        torch.cuda.empty_cache()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}]')

print('Finished')

torch.save(model.state_dict(), 'unet.pth')