import torch
import torch.nn.functional as F

#ConvNet for hyperparameter search
class HPConvNet(torch.nn.Module):
    def __init__(self,config, num_classes):
        super(HPConvNet, self).__init__()

        self.num_classes = num_classes
        self.num_conv_layers = config['num_conv'] #num of conv layers besides the initial input layer
        self.num_dense_layers = config['num_dense'] #num of FC layers after the CNN layers
        self.conv = [None for _ in range(self.num_conv_layers+1)]
        self.dense = [None for _ in range(self.num_dense_layers)]
        self.dropout = [None for _ in range(self.num_dense_layers+1)]

        self.conv[0] = torch.nn.Conv2d(1,config['cnn_0'],kernel_size=3) #input CNN layer
        self.dropout[0] = torch.nn.Dropout(config[f'dropout_0'])

        def calc_conv_out(width,kernel_size=3,stride=1,padding=0,dilation=1):
            return int((width + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1)
        def calc_maxpool(width,kernel_size=2,stride=2):
            return int((width-2)/2 +1)


        cnn_width = calc_conv_out(28) #28 is the dimension of MNIST input
        cnn_width = calc_maxpool(cnn_width) #max pooling output size
        #following CNN layers
        for i in range(self.num_conv_layers):
            cnn_width = calc_maxpool(calc_conv_out(cnn_width)) #must divide by 2 because of max pooling
            self.conv[i+1] = torch.nn.Conv2d(config[f'cnn_{i}'],config[f'cnn_{i+1}'],kernel_size=3)
        
        #CNN output size after poolings and flatten 
        after_flatten_size = cnn_width*cnn_width*config[f'cnn_{self.num_conv_layers}'] #width*height*(output dim of last CNN layer)


        #following FC layers
        for i in range(self.num_dense_layers):
            dense_size = after_flatten_size if(i==0) else config[f'dense_{i}']
            self.dense[i]= torch.nn.Linear(dense_size, config[f'dense_{i+1}'])
            self.dropout[i+1] = torch.nn.Dropout(config[f'dropout_{i+1}'])

        #output layer
        out_size = config[f'dense_{self.num_dense_layers}'] if self.num_dense_layers > 0 else after_flatten_size #in case there were no FC layer after the CNNs
        self.dense_out = torch.nn.Linear(out_size, num_classes)
    
    def forward(self,input):
        
        x = F.max_pool2d(F.relu(self.conv[0](input)),2)

        for i in range(1,self.num_conv_layers+1):
            x = F.max_pool2d(F.relu(self.conv[i](x)),2)
            
        x = self.dropout[0](x)
        x = torch.flatten(x, 1)

        for i in range(1,self.num_dense_layers+1):
            x = self.dense[i-1](x)
            x = self.dropout[i](x)
        
        x = self.dense_out(x)

        return x
