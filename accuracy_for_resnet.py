import torch.nn as nn
from torch.nn import Sequential
import torch.optim as optim
# now going to define the optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# now going to train the model 
num_epochs = 10
train_losses, train_acc_list, test_acc_list = [], [], []
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=1)

for epoch in range(num_epochs):
    model1.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
      correct = 0
      total_loss = 0
      size = 0
      for inputs, labels in testloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model1(inputs)
        loss = criterion(outputs,labels)
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).float().sum().item()
        size += labels.size(0)
    accuracy = correct / size
    print(f"Test Error:\n Accuracy: {(100 * accuracy):>0.1f}%\n")