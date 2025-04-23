import torch
import numpy as np
from params import settings
from data_prepro import load_data
from train import train_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parameters setting
args = settings()
args.epochs = 2

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_s, data_f, train_loader_list, test_loader_list = load_data(args, n_splits=2)

save_model_dir = f"../trained_models"
os.makedirs(save_model_dir, exist_ok=True)

for fold, (train_loader, test_loader) in enumerate(zip(train_loader_list, test_loader_list)):
    print(f"Training on fold {fold+1}")
    model = train_model(data_s, data_f, train_loader, test_loader, args, fold+1)

    model_path = os.path.join(save_model_dir, f"LDA_fold{fold+1}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model for fold {fold+1} saved to {model_path}")
