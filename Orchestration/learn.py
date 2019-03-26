import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from Orchestration.model import RNNModel
from Orchestration.get_data import get_train_data

def train():
    X, y = get_train_data()
    features_train, features_test, targets_train, targets_test = train_test_split(X,   
                                                                             y,
                                                                             test_size = 0.2)
    batch_size = 100
    n_iters = 2500
    num_epochs = n_iters / (len(features_train) / batch_size)
    num_epochs = int(num_epochs)

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
    test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
        
    # Create RNN
    input_dim = 128    # input dimension
    hidden_dim = 100  # hidden layer dimension
    layer_dim = 2     # number of hidden layers
    output_dim = 10   # output dimension

    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

    # Cross Entropy Loss 
    error = nn.CrossEntropyLoss()

    # SGD Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    seq_dim = 28  
    loss_list = []
    iteration_list = []
    accuracy_list = []
    count = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            train  = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels )
                
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(train)
            
            # Calculate softmax and ross entropy loss
            loss = error(outputs, labels)
            
            # Calculating gradients
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            count += 1
            
            if count % 250 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images.view(-1, seq_dim, input_dim))
                    
                    # Forward propagation
                    outputs = model(images)
                    
                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]
                    
                    # Total number of labels
                    total += labels.size(0)
                    
                    correct += (predicted == labels).sum()
                
                accuracy = 100 * correct / float(total)
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                if count % 500 == 0:
                    # Print Loss
                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))