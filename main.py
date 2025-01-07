# PyTorch for neural network and tensor operations
 import torch
 import torch.nn as nn
 import torch.optim as optim
 from torchvision import datasets, transforms
 from torch.utils.data import DataLoader

 # Matplotlib for visualizing data and results
 import matplotlib.pyplot as plt

 import matplotlib.pyplot as plt  # For plotting images
 from torchvision import datasets  # For loading the MNIST dataset
 # Download and load the MNIST dataset
 # `root='./data'` specifies where to store the dataset
 # `train=True` loads the training set (set to False for the test set)
# `download=True` ensures the dataset is downloaded if not already present
 mnist_data = datasets.MNIST(root='./data', train=True, download=True)
 # Function to display a specified number of images from the dataset
 def show_images(dataset, num_images=5):
    # Create a figure with a row of subplots
    # `figsize=(10, 2)` controls the size of the entire figure
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    
    # Loop through the first `num_images` items in the dataset
    for i in range(num_images):
        # Retrieve the image and label for the current item
        image, label = dataset[i]
        
        # Display the image on the i-th subplot
        # `cmap="gray"` displays the image in grayscale
        axes[i].imshow(image, cmap="gray")
        
        # Set the title for the current subplot to show the label
        axes[i].set_title(f"Label: {label}")
        
        # Turn off axis markers for a cleaner look
        axes[i].axis("off")
    
    # Show the figure with all images
    plt.show()
 # Call the function to display 5 items from the dataset
 show_images(mnist_data, 5)

 #Complete the code below
 # Import necessary libraries for data visualization and manipulation
 import matplotlib.pyplot as plt # For plotting and visualizing images
 import numpy as np # For numerical operations, if needed
 # Import PyTorch libraries for data handling and transformations
 from torchvision import datasets, transforms  # For loading and transforming datasets
 from torch.utils.data import DataLoader  # For creating data loaders to handle batches
 #####################################################################
 # Data preprocessing transformations: Declare the variable transform that:
 # -Convert images to PyTorch tensors, 
# -Normalize to range [-1, 1]
 #complete the line below
 resize = (28, 28) 
 transform = transforms.Compose([ 
    transforms.Resize(resize), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
 #####################################################################
 #####################################################################
 #Define the Dataset.MNIST variables train_dataset, testdataset with parameters
 # - root='./data': Saves the dataset to a local 'data' directory
 # - train=True: Loads the training set (60,000 images)
 # - download=True: Downloads the dataset if it is not already in the specified directory
 # - transform=transform: Applies the defined transformations to the dataset
 # Download and load MNIST training and test datasets
 
 train_dataset = datasets.MNIST( root='./data', train=True, 
download=True, transform=transform)
 test_dataset = datasets.MNIST( root='./data', train=False, 
download=True, transform=transform )
 #####################################################################
 #####################################################################
 # DataLoader for batching and shuffling
 batch_size = 64  # Number of images per batch
 #Define two Dataloader variables: train_loader and test_loader
 #Parameters of the DataLoader call for training:
 # - batch_size=64: Defines the number of images in each batch
 # - shuffle=True: Shuffles the training dataset for better learning (randomizes each epoch)
 # the variable will take its data from the train_dataset variable defined above
 train_loader = DataLoader(train_dataset, batch_size=batch_size, 
shuffle=True)
 #Parameters of the DataLoader call for testing:
 # - batch_size=64: Defines the number of images in each batch
 # - shuffle=false: Does not shuffle the test dataset to maintain consistency during evaluation
 #Complete the line below
 # the variable will take its data from the test_dataset variable defined above
 test_loader =DataLoader(test_dataset, batch_size=batch_size, 
shuffle=False)
 #####################################################################
 # Retrieve the first batch of train_loader and store it as images and labels 
 examples = iter(train_loader)   # Create an iterator for the training data called examples
 images, labels = next(examples)   # Get the first batch of images and labels
 #print the shapeof the tensors `images` and `labels`.
 print("shape of tensor images: ", images.shape)
 print("shape of tensor labels: ", labels.shape)
 #####################################################################
 #####################################################################
 # Display a few images from `images` with their `labels`
 for i in range(6):
  plt.subplot(2, 3, i+1)  # Create a 2x3 grid of subplots
 plt.imshow(images[i][0], cmap='gray')  # Show the image (only 1 channel for grayscale)
 plt.title(f'Label: {labels[i].item()}')  # Display the label
 plt.axis('off')
 plt.show()
 #####################################################################
 # Define the linear model
 input_size = 28 * 28  # Each image is 28x28 pixels
 num_classes = 10  # Digits 0-9
 model = nn.Sequential(
 nn.Flatten(),
 nn.Linear(784,10)
 )  # Flatten each 28x28 image into a 784-dimensional vector
 # Single linear layer with 10 output nodes
 # Initialize weights with a small Gaussian noise, biases as zero
 print("Model structure:\n", model)
 # Cross-entropy loss function (common for classification tasks) renamed as criterion
 criterion = nn.CrossEntropyLoss()
 # Define the lr value and Stochastic Gradient Descent (SGD) optimizer
 learning_rate = 0.01  # The step size for weight updates
 optimizer = optim.SGD(model.parameters(),lr=learning_rate)
 epochs = 5  # Number of times to iterate over the entire dataset (full training process)
 accuracies = []  # List to store accuracy values for each epoch
learning_rate = 0.01  # The step size for weight updates
 #####################################################################
 #####################################################################
 #We Reinitialize Model Parameters and we reset the Optimizer
 #so that we can rerun this code with different parameters.
 #Before starting a new training run, reinitialize the model's 
parameters to random values (as they were during the first 
initialization).
 torch.nn.init.normal_(model[1].weight, mean=0.0, std=0.01)
 torch.nn.init.constant_(model[1].bias, 0)
 # Reset Optimizer: Recreate the optimizer with the new learning rate 
and ensure it uses the reinitialized model's parameters.
 optimizer = optim.SGD(model.parameters(),lr=learning_rate)
 #####################################################################
 #####################################################################
 #We train now
 criterion = nn.CrossEntropyLoss()
 for epoch in range(epochs):  # Loop over each epoch
  total_loss = 0  # Initialize the cumulative loss for the current epoch
 correct = 0  # Initialize the count of correctly predicted labels for the epoch
 total = 0  # Initialize the total number of labels processed in the epoch
 # Iterate over the data loader, which provides batches of images and their corresponding labels
 for images, labels in train_loader:
 # Flatten the 28x28 image tensors into 1D vectors of size 784
  images = images.view(images.size(0), -1)   
# Perform a forward pass through the model to get output predictions
 outputs = model(images)
 # Compute the loss using the defined criterion (e.g., CrossEntropyLoss)
 loss = criterion(outputs,labels)
 # Clear any previously accumulated gradients in the optimizer
 optimizer.zero_grad()
 # Perform backpropagation to compute gradients of the loss w.r.t. model parameters
 loss.backward()
 # Update the model parameters using the computed gradients
 optimizer.step()
 # Track training metrics
 total_loss += loss.item()  # Add the loss for this batch to the cumulative loss
 # Determine the predicted class with the highest probability
 _, predicted = outputs.max(1)  # `max(1)` returns the max value and its index for each sample along dim 1
  # Count the total number of labels in the batch
 total += labels.size(0)
        
 # Count how many predictions match the true labels
 correct += (predicted == labels).sum().item()
# After all batches in the epoch, calculate accuracy
 accuracy = 100 * correct / total  # Convert accuracy to a percentage
 accuracies.append(accuracy)  # Store accuracy for plotting
    
    # Print the metrics for the current epoch
 print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, 
Accuracy: {accuracy:.2f}%')
 #####################################################################
 # Plot epochs vs accuracy
 plt.figure(figsize=(6, 4))
 plt.plot(range(1, epochs + 1), accuracies, marker='o', linestyle='-', 
label='Accuracy')
 plt.xlabel('Epochs')
 plt.ylabel('Accuracy (%)')
 plt.title('Epochs vs Accuracy')
 plt.grid(True)
 plt.legend()
 plt.show()
 #####################################################################
# Evaluation on the test set
 # Switch the model to evaluation mode to ensure proper behavior during evaluation
 # (e.g., disables dropout and batch normalization updates).
 model.eval()
 # Disable gradient computation since we are only evaluating the model and not updating weights.
 # This reduces memory usage and speeds up the computation.
 with torch.no_grad():  
    correct = 0  # Initialize the count of correct predictions
    total = 0  # Initialize the total number of samples in the test set
    # Loop through the test data loader, which provides batches of images and labels.
    for images, labels in test_loader:
        # Flatten each 28x28 image into a 1D vector of size 784 to match the input shape expected by the model.
        images = images.view(images.size(0), -1)   
        
        # Perform a forward pass to obtain model predictions for the batch of images.
        outputs = model(images)
        
        # `outputs.max(1)` returns the maximum value and its index for each row in the batch (dim=1).
        # The index corresponds to the predicted class (e.g., 0-9 for digits in MNIST).
        _, predicted = outputs.max(1)
        
        # Count the number of labels in the current batch and add it to the total count.
        total += labels.size(0) 
# Compare the predicted labels to the true labels and count the number of correct predictions.
 correct += (predicted == labels).sum().item()
 # Calculate the overall accuracy as a percentage.
 # Accuracy = (Number of correct predictions / Total number of samples) * 100
 accuracy = 100 * correct / total
 # Print the test accuracy in percentage format.
 print(f'Test Accuracy: {accuracy:.2f}%')
 # Display predictions for a few images
 examples = iter(test_loader)         

 images,labels = next(examples)                               
# Create an iterator for the test first batch of images and labels
 #images = images.flatten(start_dim=1)                             # 
 # Flatten images, tried many times usin view/reshape/flatten bit did not work
 model.eval()                               
 
 with torch.no_grad():
  outputs=model(images)
 predictions=outputs.max(1)[1]                                       
# Get index of max probability for each image
 # Forward pass for predictions
# Get the 
# Display a few images with their predictions and true labels
 for i in range(6):
plt.subplot(2, 3, i+1)
plt.imshow(images[i].squeeze(), cmap='gray')  # Show the image added squueze instead of images[i][0]
plt.title(f'Actual: {labels[i].item()}, Predicted: 
{predictions[i].item()}')
plt.axis('off')
plt.show()