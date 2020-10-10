# coding: utf-8

# In[1]:


#importing data to train_data and test_data
import torch
from torchvision import datasets
import numpy as np
import torchvision.transforms as transforms
import mlflow
import mlflow.sklearn

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./Data',train = 'True',download = True,transform = transform)
test_data = datasets.MNIST(root='./Data',train = 'False',download = True,transform = transform)


# In[2]:


#Batch_size reduced to 1( no more batch processing because of high loss with batch size 100 and 20)
num_workers = 0
batch_size = 1

train_loader = torch.utils.data.DataLoader(train_data,batch_size,num_workers)
test_loader = torch.utils.data.DataLoader(test_data,batch_size,num_workers)


# In[3]:


#Verify the training data set
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

dataiter = iter(train_loader)
images,label = dataiter.next()
images = images.numpy()

single_image = np.squeeze(images[0])

fig = plt.figure(figsize=(5,5))
axis = fig.add_subplot(111)
axis.imshow(single_image,cmap='gray')


# In[4]:


#Main network
#Step1 - Conv. layer on input 28*28*1 to transform it to 28*28*16
#Step2 - Max pool layer for pixel reducation from 28*28*16 to 14*14*16
#Step3 - Conv. layer on input 14*14*16 to transform it to 14*14*2
#Step4 - Max pool layer for pixel reducation from 14*14*2 to 7*7*2
#Step5 - Flatten 7*7*2 to 98 pixels array
#Step6 - Linear layer to transform from 98 to 80 pixel
#Step7 - For classification use pixels from 70 till 80 and use these 10 pixels as classifer output
#        (using softmax gave huge lose)
#Step8 - Linear transform to change from 80 to 98 pixel
#Step9 - Reshape the pixels to 7*7*2
#Step10 - Transpose convolution to transfrom from 7*7*2 to 14*14*2
#Step11 - Transpose convolution to transfrom from 7*7*2 to 14*14*16
#Step12 - Transpose convolution to transfrom from 14*14*16 to 28*28*1
#          Final Step which produces the autoencoder output image

import torch.nn as nn
import torch.nn.functional as F 

class ConvAutoEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(7*7*2, 80)
        self.linear2 = nn.Linear(80, 7*7*2)
        self.t_conv1 = nn.ConvTranspose2d(2, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
    def forward(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = self.pool(xb)
        xb = F.relu(self.conv2(xb))
        xb = self.pool(xb)
        xb = xb.reshape(-1, 7*7*2)
        xb = self.linear(xb)
        #out = xb[:,70:80]
        out = F.softmax(xb[:,70:80],dim =1)
        xb = self.linear2(xb)
        xb = xb.reshape(-1,2,7,7)
        xb = F.relu(self.t_conv1(xb))
        xb = F.sigmoid(self.t_conv2(xb))

        return out,xb

Convencoder = ConvAutoEnc()
print(Convencoder)       


# In[6]:


Convencoder = ConvAutoEnc()
Convencoder.load_state_dict(torch.load('MNIST_Priori1.pth',map_location='cpu'))
Convencoder.state_dict()


# In[39]:



#MSE loss function for the autoencoder final image output
loss_function_MSE = nn.MSELoss()
#cross entropy loss function for the image classification
loss_function_CE = F.cross_entropy
#Adams optimizer
optimise = torch.optim.Adam(Convencoder.parameters(),lr=0.001)


# In[40]:


#manual loss


# In[42]:


n_epochs = 1

for epoch in range(1,n_epochs+1):
    loss_value = 0.0
    
    for data in train_loader:
        images,labels = data

        optimise.zero_grad()
        out,final = Convencoder(images)
        loss_Classify = (- out[:,labels].log().mean())
        loss_AE = loss_function_MSE(final,images)
       # loss_Classify = loss_function_CE(out,labels)
        #Total loss
        loss = loss_AE + loss_Classify
        loss.backward()
        optimise.step()
        loss_value += loss.item()*images.size(0)
    loss_value =  loss_value/len(train_loader)
    
    print('Epoch: {}   Loss value:{:.6f}'.format(epoch,loss_value))


# In[43]:


torch.save(Convencoder.state_dict(), 'MNIST_Priori2.pth')


# In[44]:


n_epochs = 1

for epoch in range(1,n_epochs+1):
    loss_value = 0.0
    
    for data in train_loader:
        images,labels = data

        optimise.zero_grad()
        out,final = Convencoder(images)
        loss_Classify = (- out[:,labels].log().mean())
        loss_AE = loss_function_MSE(final,images)
       # loss_Classify = loss_function_CE(out,labels)
        #Total loss
        loss = loss_AE + loss_Classify
        loss.backward()
        optimise.step()
        loss_value += loss.item()*images.size(0)
    loss_value =  loss_value/len(train_loader)
    
    print('Epoch: {}   Loss value:{:.6f}'.format(epoch,loss_value))


# In[45]:


torch.save(Convencoder.state_dict(), 'MNIST_Priori3.pth')


# In[46]:


Convencoder = ConvAutoEnc()
Convencoder.load_state_dict(torch.load('MNIST_Priori1.pth',map_location='cpu'))
Convencoder.state_dict()


# In[47]:


Correct_output = 0
Incorrect_output = 0


# In[48]:


#Trainning accuracy for classifer
for data in train_loader:
    images,labels = data
    out,final = Convencoder(images)
    _, preds  = torch.max(out, dim=1)
    if preds[0] == labels:
        Correct_output += 1
    else:
        Incorrect_output += 1
print("No. of proper prediction:"+ str(Correct_output))    
print("No. of wrong prediction:"+ str(Incorrect_output))
print("Total number of samples:"+ str(Correct_output+Incorrect_output))
print("Acurracy:", str(Correct_output/(Correct_output+Incorrect_output)))
        


# In[49]:


Correct_output_test = 0
Incorrect_output_test = 0


# In[50]:


#Testing accuracy for classifer
for data in test_loader:
    images,labels = data
    out,final = Convencoder(images)
    _, preds  = torch.max(out, dim=1)
    if preds[0] == labels:
        Correct_output_test += 1
    else:
        Incorrect_output_test += 1
        
print("No. of proper prediction:"+ str(Correct_output_test))    
print("No. of wrong prediction:"+ str(Incorrect_output_test))
print("Total number of samples:"+ str(Correct_output_test+Incorrect_output_test))
print("Acurracy:", str(Correct_output_test/(Correct_output_test+Incorrect_output_test)))
Acc=str(Correct_output_test/(Correct_output_test+Incorrect_output_test))
    
mlflow.log_metric("Loss", loss_value)
mlflow.log_metric("Accuracy", Acc)   


# In[51]:


iterate = iter(test_loader)


# In[52]:


images,label = iterate.next()
#images = images.cuda()
final,output = Convencoder(images)
print(final)
_, preds  = torch.max(final, dim=1)
print(preds[0].item())
images = images.cpu()
images =images.numpy()
output = output.view(batch_size, 1, 28, 28)
output = output.cpu()
output= output.detach().numpy()
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(25,4))


for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# In[53]:


#Netrok with hardcoded value for encoded part
import torch.nn as nn
import torch.nn.functional as Function

class ConvAutoEnc1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(7*7*2, 80)
        self.linear2 = nn.Linear(80, 7*7*2)
        self.t_conv1 = nn.ConvTranspose2d(2, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
    def forward(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = self.pool(xb)
        xb = F.relu(self.conv2(xb))
        xb = self.pool(xb)
        xb = xb.reshape(-1, 7*7*2)
        xb = self.linear(xb)
        xb[:,70:80] = F.softmax(xb[:,70:80],dim =1)
        out = xb[:,70:80]
        # Set Prior value here

        xb = self.linear2(xb)
        xb = xb.reshape(-1,2,7,7)
        xb = F.relu(self.t_conv1(xb))
        xb = F.sigmoid(self.t_conv2(xb))

        return out,xb

Convencoder1 = ConvAutoEnc1()
print(Convencoder)       


# In[55]:


model0 = ConvAutoEnc()
model0.load_state_dict(torch.load('MNIST_Priori1.pth',map_location='cpu'))
model0.state_dict()


# In[54]:


# Mode 1
iterate = iter(test_loader)


# In[55]:


# Mode 1
images,label = iterate.next()
#images = images.cuda()
final,output = model0(images)
print(final)
_, preds  = torch.max(final, dim=1)
print(preds[0].item())
images = images.cpu()
images =images.numpy()
output = output.view(batch_size, 1, 28, 28)
output = output.cpu()
output= output.detach().numpy()
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(25,4))


for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# In[59]:


# Mode 6
iterate = iter(test_loader)


# In[64]:


# Mode 1
images,label = iterate.next()
#images = images.cuda()
final,output = model6(images)
_, preds  = torch.max(final, dim=1)
print(preds[0].item())
images = images.cpu()
images =images.numpy()
output = output.view(batch_size, 1, 28, 28)
output = output.cpu()
output= output.detach().numpy()
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(25,4))


for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# In[67]:


# Mode 8
iterate = iter(test_loader)


# In[72]:


# Mode 8
images,label = iterate.next()
#images = images.cuda()
final,output = model8(images)
_, preds  = torch.max(final, dim=1)
print(preds[0].item())
images = images.cpu()
images =images.numpy()
output = output.view(batch_size, 1, 28, 28)
output = output.cpu()
output= output.detach().numpy()
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(25,4))


for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

