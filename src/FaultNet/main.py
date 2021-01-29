import os
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torch.utils.data as data_utils

from model import CNN
from feature import mean, median, sig_image

def build_argparser():

    parser = ArgumentParser()
    
    parser.add_argument("--data_path", help="path to dataset *.npy", required=True, type=str)
    parser.add_argument("--label_path", help="path to label *.npy", required=True, type=str)
    parser.add_argument("--save_model_path", help="path to saved model", default=None, type=str)
    parser.add_argument("--train_batch_size", help="num of training batch size", default=128, type=int)
    parser.add_argument("--test_batch_size", help="num of testing batch size", default=1024, type=int)
    parser.add_argument("--num_epochs", help="num of training epochs", default=100, type=int)
    
    return parser

if __name__ == "__main__":
    
    print("+++ FaultNet Initialized +++")
    
    args = build_argparser().parse_args()
    
    data_path = args.data_path
    label_path = args.label_path
    save_model_path = args.save_model_path
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    num_epochs = args.num_epochs
    
    assert os.path.exists(data_path)
    assert os.path.exists(label_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # fix results: test accuracy 98.04%
    random_state = 777
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    
    print("+++ Load Dataset +++")
    # Load Data
    data = np.load(data_path)
    labels = np.load(label_path)
    x=data[:,0:1600]
    # Feature Generation
    channel_mean=(mean(x,10)).astype(np.float32)
    x_m=sig_image(channel_mean,40)
    channel_median=(median(x,10)).astype(np.float32)
    x_md=sig_image(channel_median,40)
    x_n=sig_image(x,40)
    
    #print(x_n[0,0])
    #print(x_m[0,0])
    #print(x_md[0,0])
    
    X=np.stack((x_n,x_m,x_md),axis=1).astype(np.float32)
    #print(X.shape)
    
    print("+++ Build Dataset +++")
    trainx, testx, trainlabel, testlabel = train_test_split(X, labels.astype(np.float32), test_size=0.2, random_state=random_state)
    sig_train, sig_test = trainx,testx
    lab_train, lab_test = trainlabel,testlabel
    
    sig_train = torch.from_numpy(sig_train)
    sig_test = torch.from_numpy(sig_test)
    lab_train= torch.from_numpy(lab_train)
    lab_test = torch.from_numpy(lab_test)
    
    train_tensor = data_utils.TensorDataset(sig_train, lab_train) 
    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = train_batch_size, shuffle = True)
    test_tensor = data_utils.TensorDataset(sig_test, lab_test) 
    test_loader = data_utils.DataLoader(dataset = test_tensor, batch_size = test_batch_size, shuffle = False)
    
    #print(sig_train.size())
    print("+++ Build Model +++")
    cnn = CNN().to(device)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    total_step = len(train_loader)
    
    print("+++ Train Model +++")
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (signals, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # Run the forward pass
            signals=signals.to(device)
            labels=labels.to(device)
            outputs = cnn(signals)
            
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels.long())            
            loss_list.append(loss.item())
            
            # Backprop and perform Adam optimisation
            
            loss.backward()
            optimizer.step()
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.long()).sum().item()
            acc_list.append(correct / total)

            if (epoch+1) % 5 == 0 or epoch==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
    
    print("+++ Test Model +++")
    total_step = len(test_loader)
    #print(total_step)
    loss_list_test = []
    acc_list_test = []
    cnn.eval()
    
    with torch.no_grad():
        for i, (signals, labels) in enumerate(test_loader):
            # Run the forward pass
            signals=signals.to(device)
            labels=labels.to(device)
            outputs = cnn(signals)
            loss = criterion(outputs, labels.long())
            loss_list_test.append(loss.item())
            if epoch%10 ==0:
                print(loss)
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.long()).sum().item()
            acc_list_test.append(correct / total)
            if (epoch) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
    
    print("+++ Save Model +++")
    # Save model
    if save_model_path:
        torch.save(cnn, save_model_path)                          