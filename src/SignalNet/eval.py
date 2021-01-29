import os
import torch
import numpy as np
from model import CNN
from dataset import DatasetFolder, npy_loader

if __name__ == "__main__":
    
    print("+++ SignalNet Initialized +++")
    
    data_path = "../../data/MAFAULDA_XX"
    save_model_path = "../../data/SignalNet_demo.pth"
    train_batch_size = 1024
    test_batch_size = 1024
    val_ratio = 0.2
    
    assert os.path.exists(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random_state = 777
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
    
    print("+++ Build Model +++")
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(save_model_path, device))
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