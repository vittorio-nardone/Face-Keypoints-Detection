## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## DONE: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #
        # Conv layer output size
        # Ow = (Iw - Fw + 2P) / Sw + 1
        # Oh = (Ih - Fh + 2P) / Sh + 1
        #
        #Iw, Ih = 224, 224 (input image size)
        #Ow, Oh = Iw, Ih

        self.conv1 = nn.Conv2d(1, 32, 5) # S=1, P=0
        #Ow, Oh = (Ow - 5 + 2*0) // 1 + 1, (Oh - 5 + 2*0) // 1 + 1

        self.pool1 = nn.MaxPool2d(2)
        #Ow, Oh = Ow // 2, Oh // 2

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 48, 5) # S=1, P=0
        #Ow, Oh = (Ow - 5 + 2*0) // 1 + 1, (Oh - 5 + 2*0) // 1 + 1

        #self.pool2 = nn.MaxPool2d(2)
        #Ow, Oh = Ow // 2, Oh // 2

        self.bn2 = nn.BatchNorm2d(48)
        
        self.conv3 = nn.Conv2d(48, 56, 5) # S=1, P=0
        #Ow, Oh = (Ow - 5 + 2*0) // 1 + 1, (Oh - 5 + 2*0) // 1 + 1

        #self.pool3 = nn.MaxPool2d(2)
        #Ow, Oh = Ow // 2, Oh // 2

        self.bn3 = nn.BatchNorm2d(56)

        self.conv4 = nn.Conv2d(56, 64, 3) # S=1, P=0
        #Ow, Oh = (Ow - 3 + 2*0) // 1 + 1, (Oh - 3 + 2*0) // 1 + 1

        #self.pool4 = nn.MaxPool2d(2)
        #Ow, Oh = Ow // 2, Oh // 2

        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3) # S=1, P=0
        #Ow, Oh = (Ow - 3 + 2*0) // 1 + 1, (Oh - 3 + 2*0) // 1 + 1

        #self.linear1 = nn.Linear(Ow * Oh * 64, 512)
        self.linear1 = nn.Linear(5184, 512)

        self.linear2 = nn.Linear(512, 136)

    def forward(self, x):
        ## DONE: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = F.relu(self.pool1(x))

        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(self.pool1(x))

        x = self.bn2(x)

        x = self.conv3(x)
        x = F.relu(self.pool1(x))

        x = self.bn3(x)

        x = self.conv4(x)
        x = F.relu(self.pool1(x))

        x = self.bn4(x)

        x = F.relu(self.conv5(x))

        ## Flat
        x = x.view(x.size(0), -1)

        ## Dense
        x = F.relu(self.linear1(x))

        x = self.linear2(x)

        return x


if __name__ == "__main__":
    net = Net()
    print(net)
