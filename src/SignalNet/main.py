import os
import torch
import numpy as np
from model import CNN
from dataset import DatasetFolder, npy_loader

from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser

def build_argparser():

    parser = ArgumentParser()
    
    parser.add_argument("--data_path", help="path to dataset *.npy", required=True, type=str)
    parser.add_argument("--save_model_path", help="path to saved model", default=None, type=str)
    parser.add_argument("--train_batch_size", help="num of training batch size", default=1024, type=int)
    parser.add_argument("--test_batch_size", help="num of testing batch size", default=1024, type=int)
    parser.add_argument("--num_epochs", help="num of training epochs", default=25, type=int)
    parser.add_argument("--val_ratio", help="validation data size to training data size ratio", default=0.2, type=float)
    parser.add_argument("--random_state", help="num of random_state", default=777, type=int)
    
    return parser

if __name__ == "__main__":
    
    print("+++ SignalNet Initialized +++")
    
    args = build_argparser().parse_args()
    #data_path = "../../data/MAFAULDA_XX"
    #save_model_path = "../../data/SignalNet_demo.pth"
    #train_batch_size = 1024
    #test_batch_size = 1024
    #num_epochs = 25
    #val_ratio = 0.2
    
    data_path = args.data_path
    save_model_path = args.save_model_path
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    num_epochs = args.num_epochs
    val_ratio = args.val_ratio
    
    assert os.path.exists(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # fix results: test accuracy 98.04% for CWRU
    # fix results: test accuracy 100% for MAFAULDA_LITE
    
    #random_state = 777
    random_state = args.random_state
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    
    print("+++ Load Dataset +++")
    dataset = DatasetFolder(
              root=data_path,
              loader=npy_loader,
              extensions='.npy'
              )
    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*val_ratio), int(len(dataset)*val_ratio)])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    
    #for x, y in test_loader:
    #    print(np.unique(y.numpy()))
        
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
            signals=signals.float().to(device)
            labels=labels.float().to(device)
            
            outputs = cnn(signals)
            
            #print(outputs.shape, labels.shape)
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
            signals=signals.float().to(device)
            labels=labels.float().to(device)
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
        torch.save(cnn.state_dict(), save_model_path)         