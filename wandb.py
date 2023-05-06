import wandb
import torchvision
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.models import resnet50
from torchsummary import summary
wandb.init()

model = resnet50(pretrained = True)

'''

.train()
.log()
.Table()
.init()
.finish()
.require()
.setup()

'''


model.train()

for batch_idx, (data,target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output,target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_intervals == 0:
        wandb.log({"log": loss})


images_t = 

wandb.log({"examples" : [wandb.Image(im) for im in images_t]})

my_table = wandb.Table()
my_table.add_column("image", images_t)
my_table.add_column("label",labels)
my_table.add_column("class_prediction",predicitons_t)
wandb.log({"mnist_predicitons": my_table})
