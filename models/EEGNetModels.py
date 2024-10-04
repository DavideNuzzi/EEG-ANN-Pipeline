import torch
import torch.nn as nn
from models.BaseModel import BaseModel
from models.custom_blocks import Conv2dMaxNorm

# ---------------------------------------------------------------------------- #
#                                  ShallowNet                                  #
# ---------------------------------------------------------------------------- #
class ShallowNet(BaseModel):

    def __init__(self, num_classes, channels, filters=40):
        super(ShallowNet, self).__init__()

        self.num_classes = num_classes
        self.filters = filters
        self.channels = channels

        self.conv_1 = Conv2dMaxNorm(1, filters, kernel_size=(1, 13), max_norm_val=2)
        self.conv_2 = Conv2dMaxNorm(filters, filters, kernel_size=(self.channels, 1), bias=False, max_norm_val=2)
        self.batch_norm = nn.BatchNorm2d(filters)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 35), stride=(1, 7))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.max_pool(x)
        x = torch.log(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def loss(self, y, y_pred):
        return torch.nn.functional.cross_entropy(y_pred, y, reduction='mean') * y.shape[0]

    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        y_pred = self.forward(x)

        # Loss
        loss = self.loss(y, y_pred)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        # Accuracy predizioni
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        accuracy = torch.sum(y_pred_class == y)

        return {'loss': loss.item(), 'accuracy': accuracy.item()}

    def predict(self, x):
        
        # Forward pass
        y_pred = self(x)

        # Faccio il softmax e poi prendo l'indice con probabilità maggiore
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        return y_pred_class

# ---------------------------------------------------------------------------- #
#                                   Deep Net                                   #
# ---------------------------------------------------------------------------- #
class DeepNet(nn.Module):

    def __init__(self, num_classes, channels):
        super(DeepNet, self).__init__()

        self.num_classes = num_classes
        self.channels = channels

        conv_filters=(25, 50, 100, 200)

        blocks = []

        blocks.append(Conv2dMaxNorm(1, conv_filters[0], kernel_size=(1,5), max_norm_val=2))

        for i, filters in enumerate(conv_filters):
            if i == 0:
                kernel_size = (channels, 1)
                ch = 25
            else:
                kernel_size = (1,10)
                ch = conv_filters[i-1] 

            conv_block = nn.Sequential(
                Conv2dMaxNorm(ch, filters, kernel_size=kernel_size, max_norm_val=2),
                nn.BatchNorm2d(filters, eps=1e-5, momentum=0.9),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)),
                nn.Dropout(0.5)
            )
            blocks.append(conv_block)

        self.main_layers = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x):

        x = self.main_layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def loss(self, y, y_pred):
        return torch.nn.functional.cross_entropy(y_pred, y, reduction='mean') * y.shape[0]
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        y_pred = self.forward(x)

        # Loss
        loss = self.loss(y, y_pred)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        # Accuracy predizioni
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        accuracy = torch.sum(y_pred_class == y)

        return {'loss': loss.item(), 'accuracy': accuracy.item()}

    def predict(self, x):
        
        # Forward pass
        y_pred = self(x)

        # Faccio il softmax e poi prendo l'indice con probabilità maggiore
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        return y_pred_class
    

# ---------------------------------------------------------------------------- #
#                                    EEGNet                                    #
# ---------------------------------------------------------------------------- #
class EEGNet(nn.Module):

    def __init__(self, num_classes, channels):
        super(EEGNet, self).__init__()

        self.num_classes = num_classes
        self.channels = channels

        F1, F2 = 8, 16

        self.layers = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding='same', bias=False), # First temporal convolution
            nn.BatchNorm2d(F1),
            Conv2dMaxNorm(F1, F2, kernel_size=(channels, 1), bias=False, groups=F1, max_norm_val=1), # Depthwise convolution
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(0.5),
            nn.Conv2d(F2, F2, kernel_size=(1, 16), bias=False, groups=F2),  # Separable = Depthwise + Pointwise
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),              # Separable = Depthwise + Pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def loss(self, y, y_pred):
        return torch.nn.functional.cross_entropy(y_pred, y, reduction='mean') * y.shape[0]
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        y_pred = self.forward(x)

        # Loss
        loss = self.loss(y, y_pred)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        # Accuracy predizioni
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        accuracy = torch.sum(y_pred_class == y)

        return {'loss': loss.item(), 'accuracy': accuracy.item()}

    def predict(self, x):
        
        # Forward pass
        y_pred = self(x)

        # Faccio il softmax e poi prendo l'indice con probabilità maggiore
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        return y_pred_class