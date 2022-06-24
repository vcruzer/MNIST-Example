import torch
import torch.nn.functional as F
from utils_dataset import read_mnist
from sklearn.metrics import accuracy_score
import numpy as np

EPOCHS = 10
BATCH_SIZE = 128

class ConvNet(torch.nn.Module):
    def __init__(self,num_classes):
        super(ConvNet, self).__init__()

        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(1,32,kernel_size=3)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.dropout = torch.nn.Dropout(0.5)
        self.dense_out = torch.nn.Linear(5408, num_classes)
    
    def forward(self,input):
        
        x = self.pool(F.relu(self.conv1(input)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.dense_out(x)

        return x

def train_model(model,train_data,val_data, optimizer,patience=2):

    train_x, train_y = train_data
    val_x,val_y = torch.tensor(val_data[0]), val_data[1]

    train_dset = []; val_dset =[]
    for i in range(len(train_y)):
        train_dset.append((train_x[i],train_y[i]))

    trainloader = torch.utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE, num_workers=4)
    best_model = ConvNet(model.num_classes)

    patience_counter = 0; 
    for epoch in range(EPOCHS):
        #batch training
        for batch_id, (batch_x, batch_y) in enumerate(trainloader):

            ypred = model(batch_x)

            loss = F.cross_entropy(ypred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            
            ypred = torch.argmax(ypred,dim=1).detach().numpy()
            print(f"[Epoch: {epoch}] Train Acc:{accuracy_score(batch_y,ypred):.4f} | Loss: {loss.item()}",end='\r')
        
        print(f"[Epoch: {epoch}] Train Acc:{accuracy_score(batch_y,ypred):.4f} | Loss: {loss.item()} | ", end="")

        #validation step
        with torch.no_grad():
            ypred = model(val_x)
            ypred = torch.argmax(ypred,dim=1).detach().numpy()
        
        val_acc = accuracy_score(val_y,ypred)
        print(f'Validation Acc: {val_acc:.4f}')
        
        

        #Early stopping + rollback
        if(epoch==0):
            previous_step_acc = val_acc
            best_model.load_state_dict(model.state_dict()) #saving current best run
        else:

            patience_counter+=1
            if(val_acc <= previous_step_acc) and (patience_counter==patience):
                model.load_state_dict(best_model.state_dict()) #loading weights of the best run
                break
            elif(val_acc > previous_step_acc):
                previous_step_acc = val_acc
                patience_counter = 0
                best_model.load_state_dict(model.state_dict()) #saving current best run


def test_model(model,test_data):

    test_x,test_y = torch.tensor(test_data[0]), test_data[1] 

    with torch.no_grad():
        ypred = model(test_x)
        ypred = torch.argmax(ypred,dim=1).detach().numpy()
        print(f'Test Acc: {accuracy_score(test_y,ypred):.4f}')

if(__name__== "__main__"):

    train_data, val_data, test_data = read_mnist(ch_dim=False)

    #pytorch uses channel dim in the front
    train_data[0] = np.expand_dims(train_data[0], 1)
    val_data[0] = np.expand_dims(val_data[0], 1)
    test_data[0] = np.expand_dims(test_data[0], 1)

    num_classes  = len(set(train_data[1]))
    input_size = train_data[0].shape

    model = ConvNet(num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model,train_data,val_data,optimizer)
    test_model(model,test_data)
