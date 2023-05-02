# check PyTorch versions
# import standard PyTorch modules
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # TensorBoard support

# import torchvision module to handle image manipulation
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms

# calculate train time, writing train data to files etc.
import time
import numpy as np
import pandas as pd
import json
from IPython.display import clear_output


from efficientnet_pytorch import EfficientNet

print(torch.__version__)
print(torchvision.__version__)

#device = torch.device("cuda")
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)     # On by default, leave it here for clarity
# torch.cuda.current_device()

# Helper class, help track loss, accuracy, epoch time, run time, 
# hyper-parameters etc. Also record to TensorBoard and write into csv, json

# import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict
from collections import namedtuple
from itertools import product

# Read in the hyper-parameters and return a Run namedtuple containing all the 
# combinations of hyper-parameters
class RunBuilder():
  @staticmethod
  def get_runs(params):

    Run = namedtuple('Run', params.keys())

    runs = []
    for v in product(*params.values()):
      runs.append(Run(*v))
    
    return runs

class RunManager():

  def __init__(self):

    # tracking every epoch count, loss, accuracy, time
    self.epoch_count = 0
    self.epoch_loss = 0
    self.epoch_num_correct = 0
    
    self.epoch_val_loss = 0
    self.epoch_val_num_correct = 0
    self.epoch_start_time = None

    # tracking every run count, run data, hyper-params used, time
    self.run_params = None
    self.run_count = 0
    self.run_data = []
    self.run_start_time = None

    # record model, loader and TensorBoard 
    self.network = None
    self.loader = None
    self.vloader = None
    self.tb = None

  # record the count, hyper-param, model, loader of each run
  # record sample images and network graph to TensorBoard  
  def begin_run(self, run, network, loader,vloader):

    self.run_start_time = time.time()

    self.run_params = run
    self.run_count += 1

    self.network = network
    self.loader = loader
    self.vloader = vloader
    self.tb = SummaryWriter(comment=f'-{run}')

    images, labels = next(iter(self.loader))
    images, labels = images.to(device), labels.to(device)
    grid = torchvision.utils.make_grid(images)

    self.tb.add_image('images', grid)
    self.tb.add_graph(self.network, images)

  # when run ends, close TensorBoard, zero epoch count
  def end_run(self):
    self.tb.close()
    self.epoch_count = 0

  # zero epoch count, loss, accuracy, 
  def begin_epoch(self):
    self.epoch_start_time = time.time()

    self.epoch_count += 1
    self.epoch_loss = 0
    self.epoch_num_correct = 0

  # 
  def end_epoch(self):
    # calculate epoch duration and run duration(accumulate)
    epoch_duration = time.time() - self.epoch_start_time
    run_duration = time.time() - self.run_start_time

    # record epoch loss and accuracy
    loss = self.epoch_loss / len(self.loader.dataset)
    accuracy = self.epoch_num_correct / len(self.loader.dataset)
    
    vloss = self.epoch_val_loss / len(self.vloader.dataset)
    vaccuracy = self.epoch_val_num_correct / len(self.vloader.dataset)

    # Record epoch loss and accuracy to TensorBoard 
    self.tb.add_scalar('Loss', loss, self.epoch_count)
    self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
    
    # Record epoch loss and accuracy to TensorBoard 
    self.tb.add_scalar('vLoss', vloss, self.epoch_count)
    self.tb.add_scalar('vAccuracy', vaccuracy, self.epoch_count)

    # Record params to TensorBoard
    for name, param in self.network.named_parameters():
      self.tb.add_histogram(name, param, self.epoch_count)
      self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
    
    # Write into 'results' (OrderedDict) for all run related data
    results = OrderedDict()
    results["run"] = self.run_count
    results["epoch"] = self.epoch_count
    results["loss"] = loss
    results["val_loss"] = self.epoch_val_loss
    results["val_accuracy"] = self.epoch_val_num_correct
    results["accuracy"] = accuracy
    results["epoch duration"] = epoch_duration
    results["run duration"] = run_duration

    # Record hyper-params into 'results'
    for k,v in self.run_params._asdict().items(): results[k] = v
    self.run_data.append(results)
    df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')

    # display epoch information and show progress
    clear_output(wait=True)
    #display(df)

  def track_vloss_vacc(self, loss, acc):
    # multiply batch size so variety of batch sizes can be compared
    self.epoch_val_loss = loss.item()
    self.epoch_val_num_correct = acc
    
    
    # accumulate loss of batch into entire epoch loss
  def track_loss(self, loss):
    # multiply batch size so variety of batch sizes can be compared
    self.epoch_loss += loss.item() * self.loader.batch_size

  # accumulate number of corrects of batch into entire epoch num_correct
  def track_num_correct(self, preds, labels):
    self.epoch_num_correct += self._get_num_correct(preds, labels)

  @torch.no_grad()
  def _get_num_correct(self, preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
  
  # save end results of all runs into csv, json for further a
  def save(self, fileName):

    pd.DataFrame.from_dict(
        self.run_data, 
        orient = 'columns',
    ).to_csv(f'{fileName}.csv')

    with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
      json.dump(self.run_data, f, ensure_ascii=False, indent=4)

#set device
if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_properties(device),torch.cuda.set_device(device),torch.cuda.current_device())

#Hyperparameters

#input_size
#num_classes = 50
#learning_rate = 0.00005
BATCH_SIZE = 16
epochs = 1

# put all hyper params into a OrderedDict, easily expandable
params = OrderedDict(
    exp='6',
    lr = [0.0005],#,0.001,0.0001,0.00001], # 0.001
    batch_size = [BATCH_SIZE], # 1000
    shuffle = [True] # True,False
    # Optimizer = [Adam,NAdam,RMSProp,Adamax,SGD,Adagrad,Adadelta]
)

TRAIN_DATA_PATH = "./data_a6/train/"
TEST_DATA_PATH = "./data_a6/test/"

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)

m = RunManager()

# get all runs from params using RunBuilder class
for run in RunBuilder.get_runs(params):

    # if params changes, following line of code should reflect the changes too
    
    #network = EfficientNet.from_name('efficientnet-b7')
    network = models.resnet101(pretrained=True)
    num_ftrs = network.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    network.fc = nn.Linear(num_ftrs, 50)
    
    #network = SimpleNet()
    network= network.to(device)
    
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=run.batch_size, shuffle=run.shuffle,  num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=run.batch_size, shuffle=run.shuffle,  num_workers=4)
    #loader = torch.utils.data.DataLoader(train_data, batch_size = run.batch_size)
    #testloader = torch.utils.data.DataLoader(test_data, batch_size = 100, shuffle=False, num_workers=0)
    #optimizer = optim.SGD(network.parameters(), lr=run.lr, momentum=0.9)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in network.state_dict():
        print(param_tensor, "\t", network.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    #optimizer = run.opt(network.parameters(), lr=run.lr)
    loader = train_data_loader
    testloader = test_data_loader
    m.begin_run(run, network, loader,testloader)
    for epoch in range(epochs):
        start = time.time()
        network.train()
        m.begin_epoch()
        running_loss=0
        for i,batch in enumerate(loader,0):
            images = batch[0]
            labels = batch[1]
            images = images.to(device)
            labels = labels.to(device)
            
            #images = images.reshape(images.shape[0],-1)

            #forward
            preds = network(images)
            loss = F.cross_entropy(preds, labels) # Adam, SGD, RSPROP

            #backward
            optimizer.zero_grad()
            loss.backward()

            #gradient descent or adam step
            optimizer.step()

            running_loss+=loss.data

            if i%10==9:
                end=time.time()
                print ('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s'%(epoch+1,(i+1)*BATCH_SIZE,running_loss/100,(end-start)))
                #tb.add_scalar('Loss', loss, epoch+1)
                start=time.time()
                running_loss=0

            m.track_loss(loss)
            m.track_num_correct(preds, labels)

            #vacc, vloss = get_test(network,testloader)
            #m.track_vloss_vacc(vloss, vacc)
        m.end_epoch()    
    m.end_run()

# when all runs are done, save results to files
m.save('results')
#save model state dictionary
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

torch.save(network.state_dict(), 'models/'+current_time)
#load model state dict
#model = model() #construcor
#model.load_state_dict(torch.load('models/'+BATCH_SIZE))
#model.eval()



# since we're not training, we don't need to calculate the gradients for our outputs
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=run.batch_size, shuffle=run.shuffle,  num_workers=4)



def check_accuracy(network,test_loader):
    correct = 0
    total = 0
    predictions =[] # np.zeros(len(test_data_loader)*32)
    true_labels = []
    with torch.no_grad():
        network.eval()
        for i,data in enumerate(test_data_loader,0):
            images, labels = data
            true_labels.append(labels)
            # calculate outputs by running images through the network
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            numpy_pred = predicted.cpu().detach().numpy()
    #         print(len(numpy_pred), numpy_pred.shape, predictions.shape)
    #         predictions[i*32:(i+1)*32] = predicted.cpu().detach().numpy()
            predictions.append(numpy_pred)
    
    print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
        
check_accuracy(network,test_data_loader)
network.train()