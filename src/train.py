import json
from datetime import datetime
import sys
import os
import argparse
import warnings
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from fvcore.common.timer import Timer

from utils.utils import *
from models.kan.LBFGS import *
from torch.utils.tensorboard import SummaryWriter

import scripts.args_KAN as args_KAN,scripts.args_MLP as args_MLP

warnings.simplefilter(action='ignore', category=UserWarning)

def train(args, model, device, train_loader, optimizer, epoch, logger, start_index):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, start_index):
        data, target = todevice(data, device), todevice(target, device)

        if args.optimizer in ["adam",'sgd']:

            optimizer.zero_grad()
            output = model(data)

            if args.loss == "cross_entropy":
                losses = [F.cross_entropy(output, target)]
            elif args.loss == "mse":
                losses = [F.mse_loss(output, target)]
            else:
                raise NotImplementedError
            
            loss = 0
            for l in losses:
                loss = loss + l
            loss.backward()
            optimizer.step()

        elif args.optimizer == "lbfgs":
            def closure():
                optimizer.zero_grad()
                output = model(data)
                if args.loss == "cross_entropy":
                    losses = [F.cross_entropy(output, target)]
                elif args.loss == "mse":
                    losses = [F.mse_loss(output, target)]
                else:
                    raise NotImplementedError

                loss = 0
                for l in losses:
                    loss = loss + l

                loss.backward()
                return loss

            optimizer.step(closure)

        if batch_idx % args.log_interval == 0:

            with torch.no_grad():
                output = model(data)
                if args.loss == "cross_entropy":
                    losses = [F.cross_entropy(output, target)]
                elif args.loss == "mse":
                    losses = [F.mse_loss(output, target)]
                else:
                    raise NotImplementedError
                
                logger.add_scalar('Train/Loss', sum(losses).item(), epoch * len(train_loader) + batch_idx)

        if args.save_model and (batch_idx + 1) % args.save_model_interval == 0:
            save_path = f"{args.log_dir}/model"
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/{batch_idx + 1}.pt")

        if args.dry_run:
            break

    return model

def test(args, model, device, test_loader, epoch, logger):
    model.eval()

    if args.loss == "cross_entropy":
        
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = todevice(data, device), todevice(target, device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        
        logger.add_scalar('Test/Loss', test_loss, epoch)
        logger.add_scalar('Test/Accuracy', 100. * correct / len(test_loader.dataset), epoch)

        return 100. * correct / len(test_loader.dataset)
    
    elif args.loss == "mse":
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = todevice(data, device), todevice(target, device)
                output = model(data)
                per_sample_loss = F.mse_loss(output, target, reduction='none')
                per_sample_rmse = torch.sqrt(per_sample_loss)
                test_loss += per_sample_rmse.sum().item()

        test_loss /= len(test_loader.dataset)
        
        logger.add_scalar('Test/Loss', test_loss, epoch)
    
    else:
        raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="KAN", #required=True,
                        help='network structure')

    args, rest_args = parser.parse_known_args()
    model = args.model

    if model == 'KAN':
        args = args_KAN.get_args(rest_args)
    elif model == 'MLP':
        args = args_MLP.get_args(rest_args)
    else:
        raise NotImplementedError
    
    args.model = model
    os.chdir(args.chdir)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    randomness_control(args.seed)

    args.save_model_interval = max(args.save_model_interval , 1)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = args.log_dir + f"/{args.dataset}/{args.model}/{args.model}_{current_time}"
    os.makedirs(args.log_dir, exist_ok = True)
    
    logger = SummaryWriter(log_dir=args.log_dir)

    train_loader, test_loader, num_classes, input_size = get_loader(args, use_cuda = use_cuda)

    args.output_size = num_classes
    args.input_size = input_size

    args.activation = get_activation(args)
    if(args.model == "KAN"):
        args.kan_shortcut_function = get_shortcut_function(args)

    model = get_model(args)
    model = model.to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == "lbfgs":
        optimizer = LBFGS(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            history_size=10, 
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-32, 
            tolerance_change=1e-32, 
            tolerance_ys=1e-32)
    else:
        raise NotImplementedError

    args_dict = {key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, list, dict))}
    json_file_path = f"{args.log_dir}/args/args.json"
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    with open(json_file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    print("Training")

    fvctimer = Timer()
    for epoch in range(1, args.epochs + 1):
        if fvctimer.is_paused():
            fvctimer.resume()
        else:
            fvctimer.reset()
        train(args, model, device, train_loader, optimizer, epoch, logger, start_index = (epoch - 1) *len(train_loader))
        fvctimer.pause()
        test(args, model, device, test_loader, epoch, logger)
        
    logger.close()
    print("Finished")

if __name__ == '__main__':
    main()
