import torch

import av
print(av.__version__)
from torchvision import datasets, models, transforms
from torchvision.datasets import UCF101

ucf_data_dir = "/home/muhammadbsheikh/workspace/try/dataset/UCF-101"
ucf_label_dir = "/home/muhammadbsheikh/workspace/try/dataset/ucfTrainTestlist"
frames_per_clip = 5
step_between_clips = 1
batch_size = 32


class Dataset:
    def __init__(self,data_id):

        self.data_id = data_id

    def get_datapath(self):
        path = ''
        if self.data_id == 'ucf101':
            path = '/home/muhammadbsheikh/workspace/try/dataset/UCF-101'

        return path


tfs = transforms.Compose([
            # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (T, C, H, W) for easier convolutions
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            # rescale to the most common size
            transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
])

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

if __name__ == "__main__":
    ds = Dataset('ucf101')
    print(ds.get_datapath())
    # create train loader (allowing batches and other extras)
train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                       step_between_clips=step_between_clips, train=True, transform=tfs)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=custom_collate)
# create test loader (allowing batches and other extras)
test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                      step_between_clips=step_between_clips, train=False, transform=tfs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=custom_collate)



lr = 0.0005
epochs = 1


network = models.resnet101(pretrained=True)
num_ftrs = network.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
network.fc = nn.Linear(num_ftrs, 101)
network= network.to(device)

optimizer = optim.Adam(network.parameters(), lr=lr)

for epoch in range(epochs):
    start = time.time()
    network.train()
    running_loss=0
    for i,batch in enumerate(train_loader,0):
        images = batch[0]
        labels = batch[1]
        images = images.to(device)
        labels = labels.to(device)

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




