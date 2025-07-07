import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
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

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout5 = nn.Dropout(0.1)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout6 = nn.Dropout(0.1)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout7 = nn.Dropout(0.1)
        self.conv15 = nn.Conv2d(32, 1, kernel_size=1)

        self.last_layer = self.conv15 #Сохраняем для инициализации

      #Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.last_layer.bias is not None:
            nn.init.constant_(self.last_layer.bias, 0)
        nn.init.normal_(self.last_layer.weight, mean=0, std=0.01)

    def forward(self, x):
        conv1 = nn.functional.relu(self.conv1(x))
        conv2 = nn.functional.relu(self.conv2(conv1))
        conv2 = self.dropout1(conv2)
        pool1 = self.pool1(conv2)

        conv3 = nn.functional.relu(self.conv3(pool1))
        conv4 = nn.functional.relu(self.conv4(conv3))
        conv4 = self.dropout2(conv4)
        pool2 = self.pool2(conv4)

        conv5 = nn.functional.relu(self.conv5(pool2))
        conv6 = nn.functional.relu(self.conv6(conv5))
        conv6 = self.dropout3(conv6)
        pool3 = self.pool3(conv6)

        conv7 = nn.functional.relu(self.conv7(pool3))
        conv8 = nn.functional.relu(self.conv8(conv7))
        conv8 = self.dropout4(conv8)

          # Decoder
        up1 = self.up1(conv8)

        conv9 = nn.functional.relu(self.conv9(torch.cat([up1, conv6], dim=1)))
        conv10 = nn.functional.relu(self.conv10(conv9))
        conv10 = self.dropout5(conv10)

        up2 = self.up2(conv10)
        conv11 = nn.functional.relu(self.conv11(torch.cat([up2, conv4], dim=1)))
        conv12 = nn.functional.relu(self.conv12(conv11))
        conv12 = self.dropout6(conv12)

        up3 = self.up3(conv12)

        conv13 = nn.functional.relu(self.conv13(torch.cat([up3, conv2], dim=1)))
        conv14 = nn.functional.relu(self.conv14(conv13))
        conv14 = self.dropout7(conv14)

        conv15 = self.conv15(conv14)
        outputs = conv15
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
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
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


            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            return image, mask

        except Exception as e:
            print(f"Ошибка при обработке {img_name}: {e}")
            return None

transform = transforms.Compose([
    transforms.ToTensor(),
])




train_image_dir = os.path.join('C:/Users/Developer/PycharmProjects/Segm_test/train/images')
train_mask_dir = os.path.join('C:/Users/Developer/PycharmProjects/Segm_test/train/masks')
val_image_dir = os.path.join('C:/Users/Developer/PycharmProjects/Segm_test/val/images')
val_mask_dir = os.path.join('C:/Users/Developer/PycharmProjects/Segm_test/val/masks')


train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

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

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss = loss / accumulation_steps
        loss.backward()
        running_loss += loss.item()*accumulation_steps

        if (i + 1) % accumulation_steps == 0:
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
            optimizer.step()
            optimizer.zero_grad()

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