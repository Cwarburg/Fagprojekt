import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YOLODataset(Dataset):
    def __init__(
        self, 
        img_dir: str, 
        label_dir: str, 
        img_size: int = 416, 
        transform=None
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        
        self.image_files = [
            f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size
        
        # Load annotations
        label_path = os.path.join(
            self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.txt'
        )
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    # Convert coordinates to pixel values
                    x_center *= original_width
                    y_center *= original_height
                    w *= original_width
                    h *= original_height
                    # Convert to corner coordinates
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))
        else:
            print(f"No label found for {img_path}")
        
        # Resize the image and adjust annotations
        if self.transform:
            image = self.transform(image)
            scale_x = self.img_size / original_width
            scale_y = self.img_size / original_height
            boxes = [[
                box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y
            ] for box in boxes]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return image, target


transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])


def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images)
    targets = [{k: torch.as_tensor(v) for k, v in t.items()} for t in targets]
    return images, targets

# Replace 'NUM_CLASSES' with the number of classes in your dataset
NUM_CLASSES = 1  # Example: 20 classes

train_dataset = YOLODataset(
    img_dir= '/zhome/2e/9/187921/Desktop/fagprojectdata/train/images',
    label_dir='/zhome/2e/9/187921/Desktop/fagprojectdata/train/labels',
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataset = YOLODataset(
    img_dir='/zhome/2e/9/187921/Desktop/fagprojectdata/valid/images',
    label_dir='/zhome/2e/9/187921/Desktop/fagprojectdata/valid/labels',
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn
)


class SimpleYOLO(nn.Module):
    def __init__(self, num_classes: int = 1, grid_size: int = 13, num_bboxes: int = 2):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.bbox_attrs = 5  # 5 for bbox (x, y, w, h, conf), rest for classes
        out_channels = self.num_bboxes * self.bbox_attrs
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # [batch, 16, 416, 416]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [batch, 16, 208, 208]
            
            nn.Conv2d(16, 32, 3, 1, 1),  # [batch, 32, 208, 208]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [batch, 32, 104, 104]
            
            nn.Conv2d(32, 64, 3, 1, 1),  # [batch, 64, 104, 104]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [batch, 64, 52, 52]
            
            nn.Conv2d(64, 128, 3, 1, 1),  # [batch, 128, 52, 52]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [batch, 128, 26, 26]
            
            nn.Conv2d(128, 256, 3, 1, 1),  # [batch, 256, 26, 26]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [batch, 256, 13, 13]
            
            nn.Conv2d(256, 512, 3, 1, 1),  # [batch, 512, 13, 13]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        )
        
        # Final convolutional layer
        self.pred = nn.Conv2d(512, out_channels, 1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pred(x)  # [batch, out_channels, grid_size, grid_size]
        x = x.permute(0, 2, 3, 1)  # [batch, grid_size, grid_size, out_channels]
        return x


class YOLOLoss(nn.Module):
    def __init__(self, S=13, B=2, num_classes=1):
        super(YOLOLoss, self).__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per grid cell
        self.C = num_classes  # Number of classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    
    def compute_iou(self, box1, box2):
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4] / 2
        box1_x1 = box1_xy - box1_wh
        box1_x2 = box1_xy + box1_wh
        
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4] / 2
        box2_x1 = box2_xy - box2_wh
        box2_x2 = box2_xy + box2_wh
        
        # Intersection
        inter_x1 = torch.max(box1_x1[..., 0], box2_x1[..., 0])
        inter_y1 = torch.max(box1_x1[..., 1], box2_x1[..., 1])
        inter_x2 = torch.min(box1_x2[..., 0], box2_x2[..., 0])
        inter_y2 = torch.min(box1_x2[..., 1], box2_x2[..., 1])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        box1_area = (box1_x2[..., 0] - box1_x1[..., 0]) * \
                    (box1_x2[..., 1] - box1_x1[..., 1])
        box2_area = (box2_x2[..., 0] - box2_x1[..., 0]) * \
                    (box2_x2[..., 1] - box2_x1[..., 1])
        
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def forward(self, predictions, target):
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, self.S, self.S, self.B, -1)
        
        # Prepare target tensors
        target_tensor = torch.zeros_like(predictions)
        
        for i in range(batch_size):
            num_objs = len(target[i]['boxes'])
            for obj_idx in range(num_objs):
                obj = target[i]['boxes'][obj_idx]
                class_id = target[i]['labels'][obj_idx]

                # Map object to grid cell
                cell_size = 416 / self.S
                x_center = (obj[0] + obj[2]) / 2
                y_center = (obj[1] + obj[3]) / 2

                x = x_center / cell_size
                y = y_center / cell_size

                x = min(x, self.S - 1e-6)
                y = min(y, self.S - 1e-6)

                grid_x = int(x)
                grid_y = int(y)

                x_offset = x - grid_x
                y_offset = y - grid_y

                w = (obj[2] - obj[0]) / 416  # Normalize width
                h = (obj[3] - obj[1]) / 416  # Normalize height

                # Update target tensor
                target_tensor[i, grid_y, grid_x, 0, 0] = x_offset
                target_tensor[i, grid_y, grid_x, 0, 1] = y_offset
                target_tensor[i, grid_y, grid_x, 0, 2] = w
                target_tensor[i, grid_y, grid_x, 0, 3] = h
                target_tensor[i, grid_y, grid_x, 0, 4] = 1  # Confidence
        
        # Compute losses
        coord_mask = target_tensor[..., 4] == 1
        coord_loss = self.lambda_coord * nn.MSELoss(reduction='sum')(
            predictions[coord_mask][..., :4],
            target_tensor[coord_mask][..., :4]
        )
        
        conf_loss = nn.MSELoss(reduction='sum')(
            predictions[..., 4],
            target_tensor[..., 4]
        )
        
        class_loss = nn.MSELoss(reduction='sum')(
            predictions[coord_mask][..., 5:],
            target_tensor[coord_mask][..., 5:]
        )
        
        total_loss = coord_loss + conf_loss + class_loss
        return total_loss / batch_size


def visualize_and_save_predictions(images, outputs, targets, batch_idx, experiment_dir):
    batch_size = images.size(0)
    outputs = outputs.cpu()
    images = images.cpu()
    
    S, B = 13, 2  # Grid size and number of boxes
    
    for i in range(batch_size):
        image = images[i]
        output = outputs[i]
        target = targets[i]
        img = image.permute(1, 2, 0).numpy()
        
        output = output.view(S, S, B, 5)
        
        pred_boxes = []
        pred_scores = []
        conf_thresh = 0.1  # Lower confidence threshold to show more boxes
        for row in range(S):
            for col in range(S):
                for b in range(B):
                    conf = output[row, col, b, 4]
                    if conf > conf_thresh:
                        x_offset, y_offset, w, h = output[row, col, b, 0:4]
                        x = (col + x_offset) / S * 416
                        y = (row + y_offset) / S * 416
                        w = w * 416
                        h = h * 416
                        x_min = x - w / 2
                        y_min = y - h / 2
                        x_max = x + w / 2
                        y_max = y + h / 2
                        pred_boxes.append([x_min, y_min, x_max, y_max])
                        pred_scores.append(conf.item())
        
        gt_boxes = []
        boxes = target['boxes']
        for box in boxes:
            x_min, y_min, x_max, y_max = box.tolist()
            gt_boxes.append([x_min, y_min, x_max, y_max])
        
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        
        for j in range(len(pred_boxes)):
            box = pred_boxes[j]
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(
                box[0], box[1] - 5, f"Conf: {pred_scores[j]:.2f}",
                color='red', fontsize=8, backgroundcolor='white'
            )
        
        for box in gt_boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f'val_prediction_{batch_idx}_{i}.png'))
        plt.close(fig)


def validate(model, val_loader, criterion, experiment_dir):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            if idx % 10 == 0:  # Save predictions every 10 batches
                visualize_and_save_predictions(images, outputs, targets, idx, experiment_dir)
    
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    print("Engines ignited!")
    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("CUDA is not available.")
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model, loss function, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleYOLO(num_classes=NUM_CLASSES).to(device)
    criterion = YOLOLoss(S=13, B=2, num_classes=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create experiments directory
    experiment_dir = 'experiments/exp1'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    train_losses = []
    val_losses = []
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Validate the model
        val_loss = validate(model, val_loader, criterion, experiment_dir)
        val_losses.append(val_loss)
        
        # Plot and save the loss curves
        plt.figure()
        plt.plot(range(1, epoch+2), train_losses, label='Training Loss')
        plt.plot(range(1, epoch+2), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.savefig(os.path.join(experiment_dir, 'loss_curves.png'))
        plt.close()
    
    # Save the model after all epochs are completed
    torch.save(model.state_dict(), os.path.join(experiment_dir, 'model_final.pth'))
