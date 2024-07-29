import os
import torch

def save_model(args, model):
    torch.save(model, 'scarf_pre_train_'+args.dataset+'.pth')

def save_model_checkpoint(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch)) # there is some problem here
    # reference https://hackmd.io/@Johnsonnnn/B1OqGU6T9
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
