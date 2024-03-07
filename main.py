import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Step 1: Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# if we set the batch_size from 16 to 32 the initial loss will be below 2 which is a good sign. but the accuracy still
# remain 97%
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


# Step 2: Define the model
class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 92)
        self.fc3 = nn.Linear(92, 64)
        self.fc4 = nn.Linear(64, 10)
        # self.dropout = nn.Dropout(0.2)  # Adding dropout doesnt really improve th accuracy but decrease slightly

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)  # Applying dropout after the first fully connected layer
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Step 3: Wrap the model with DataParallel
model = nn.DataParallel(simpleNN()).to(device)

# Step 4: Define Loss Function and Optimizer

criterion = nn.CrossEntropyLoss()
# Using RMSprop will decrease the accuracy for a fair bit amount of % in this case it dropped from 97% to 89% which
# should be considered significant.
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

# this will remain the accuracy of 97% even after introducing momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Step 5: Training Loop
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Step 6: Evaluate the Model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
