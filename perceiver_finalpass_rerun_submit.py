import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchgeo.datasets import MillionAID
import torchvision.transforms as transforms
from transformers import PerceiverForImageClassificationLearned
from torch.utils.data import random_split
from transformers import AdamW
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import evaluate

# Define transformations for the data
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL Image
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert back to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Function to apply transform directly to the dataset
class CustomMillionAID(MillionAID):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data['image'] = transform(data['image'])
        return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("--------------",device)

#device='cpu'
allocated_memory=torch.cuda.memory_allocated()
print(allocated_memory)

# Loading the dataset
train_ds = CustomMillionAID(root='./data', task="multi-class", split="train")
# Define the split ratio (e.g., 80% for training, 20% for testing)
train_size = int(0.8 * len(train_ds))
test_size = len(train_ds) - train_size
# Split the dataset
train_dataset, test_dataset = random_split(train_ds, [train_size, test_size])
# Define DataLoader for the training and test sets
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)#, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

for i, data in enumerate(trainloader,0):
    print (i, data)
    break

model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned",
                                                               num_labels=51,
                                                               ignore_mismatched_sizes=True)
model.to(device)

# Directory to save the model
save_dir = "E:\danny_dl_models\saved_models"
os.makedirs(save_dir, exist_ok=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
def train(model, train_loader, optimizer, device, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(num_epochs):
        print("Epoch:", epoch)

        for batch in tqdm(trainloader):
            # get the inputs;
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # evaluate
            predictions = outputs.logits.argmax(-1).cpu().detach().numpy()
            accuracy = accuracy_score(y_true=batch["label"].numpy(), y_pred=predictions)
            #print(f"Loss: {loss.item()}, Accuracy: {accuracy}")

        # Save model after each epoch
        model_save_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_save_path)
        print(f"Model saved after epoch {epoch+1}")

# Number of epochs
num_epochs = 10

train(model, trainloader, optimizer, device, num_epochs)

accuracy = evaluate.load("accuracy")

model.eval()
for batch in tqdm(testloader):
      # get the inputs;
      inputs = batch["image"].to(device)
      labels = batch["label"].to(device)

      # forward pass
      outputs = model(inputs=inputs, labels=labels)
      logits = outputs.logits
      predictions = logits.argmax(-1).cpu().detach().numpy()
      references = batch["label"].numpy()
      accuracy.add_batch(predictions=predictions, references=references)

final_score = accuracy.compute()
print("Accuracy on test set:", final_score)

