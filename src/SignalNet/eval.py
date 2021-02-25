import os
import torch
import numpy as np
from model import CNN
from dataset import DatasetFolder, npy_loader

from torch.nn import CrossEntropyLoss
import numpy as np
from argparse import ArgumentParser

def build_argparser():

    parser = ArgumentParser()
    
    parser.add_argument("--data_path", help="path to dataset *.npy", required=True, type=str)
    parser.add_argument("--save_model_path", help="path to saved model", default=None, type=str)
    parser.add_argument("--test_batch_size", help="num of testing batch size", default=1024, type=int)
    parser.add_argument("--val_ratio", help="validation data size to training data size ratio", default=0.2, type=float)
    parser.add_argument("--random_state", help="num of random_state", default=777, type=int)
    
    return parser
    
if __name__ == "__main__":
    
    print("+++ SignalNet Initialized +++")
    
    #data_path = "../../data/MAFAULDA_XX"
    #save_model_path = "../../data/SignalNet_demo.pth"
    #test_batch_size = 1024
    #val_ratio = 0.2
    
    args = build_argparser().parse_args()
    
    data_path = args.data_path
    save_model_path = args.save_model_path
    test_batch_size = args.test_batch_size
    val_ratio = args.val_ratio
    
    assert os.path.exists(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    
    print("+++ Build Model +++")
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(save_model_path, device))
    total_step = len(test_loader)
    #print(total_step)
    #loss_list_test = []
    acc_list_test = []
    
    cnn.eval()
    criterion = CrossEntropyLoss()
    results = {}
    
    with torch.no_grad():
        for i, (signals, labels) in enumerate(test_loader):
            # Run the forward pass
            signals=signals.float().to(device)
            labels=labels.float().to(device)
            outputs = cnn(signals)
            loss = criterion(outputs, labels.long())
            #loss_list_test.append(loss.item())
            #if epoch%10 ==0:
            #    print(loss)
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.long()).sum().item()
            acc_list_test.append(correct / total)
            
            labels_arr = labels.detach().cpu().numpy()
            predict_arr = predicted.detach().cpu().numpy()
            
            for pred, true in zip(predict_arr, labels_arr):
                if true not in results:
                    results[true] = {"n_total": 1} 
                    if pred == true:
                        results[true]["n_correct"] = 1 
                    else:
                        results[true]["n_correct"] = 0
                else:
                    results[true]["n_total"] += 1 
                    if pred == true:
                        results[true]["n_correct"] += 1
            
            #print('Loss: {:.4f}, Accuracy: {:.2f}%'
            #      .format(loss.item(), (correct / total) * 100))
    
    n_total_correct = 0
    n_total_counts = 0
    for key, value in results.items():
        print("Lable: {}, Accuracy: {:.2f} %, Counts: {}".format(key, results[key]["n_correct"]/results[key]["n_total"]*100, results[key]["n_total"]))
        n_total_correct += results[key]["n_correct"]
        n_total_counts += results[key]["n_total"]
    print("Total Accuracy: {:.2f} %".format(n_total_correct/n_total_counts*100))
    
    