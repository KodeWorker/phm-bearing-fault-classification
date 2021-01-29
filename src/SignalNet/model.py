from torch.nn import Module, Conv2d, MaxPool2d, Dropout, Linear
from torch.nn.functional import relu, log_softmax

######## To print layer outputs ########
class PrintLayer(Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
                    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
########################################

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        #self.print_layer = PrintLayer()
        
        self.conv1 = Conv2d(8, 32, kernel_size=4,stride=1,padding = 1)
        self.mp1 = MaxPool2d(kernel_size=4,stride=2)
        self.conv2 = Conv2d(32,64, kernel_size=4,stride =1)
        self.mp2 = MaxPool2d(kernel_size=4,stride=2)
        self.fc1= Linear(9216,256)
        self.dp1 = Dropout(p=0.2)
        self.fc2 = Linear(256,10)

    def forward(self, x):
        in_size = x.size(0)
        x = relu(self.mp1(self.conv1(x)))
        #x = self.print_layer(x)
        x = relu(self.mp2(self.conv2(x)))
        #x = self.print_layer(x)
        x = x.view(in_size,-1)
        #x = self.print_layer(x)
        x = relu(self.fc1(x))
        #x = self.print_layer(x)
        x = self.dp1(x)
        #x = self.print_layer(x)
        x = self.fc2(x)
        
        return log_softmax(x, dim=1)