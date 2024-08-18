import math
import argparse
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from ptflops import get_model_complexity_info

from my_dataset import MyDataSet
from vit_FFN5 import vit_base_patch16_224_in21k as create_model

from utils import read_split_data, train_one_epoch

import os
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count

def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  # 确保这是你模型所需的损失函数

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)  # 计算损失
            total_loss += loss.item() * images.size(0)  # 累加损失
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(targets.view(-1).cpu().numpy())

    # 计算平均损失
    avg_loss = total_loss / len(data_loader.dataset)

    # 计算其他指标
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # 假设还需要返回accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    return avg_loss, accuracy, precision, recall, f1
def main(args):
    start_time = time.time()  # 记录开始时间                          ###1111111111
    wandb.init(project='Nuts',name='vit+CBs222222222')
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    with torch.cuda.device(0):                                                  ###22222222
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_val_acc = 0.0                   ########33333333
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)#######4444444
        # 使用wandb记录所有指标
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "precision": val_precision,
            "recall": val_recall,
            "f1_score": val_f1,
            "best_val_acc": best_val_acc
        })
        # 更新最高验证集正确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 可选：保存当前最佳模型
            torch.save(model.state_dict(), f"./weights/best_model_{epoch}.pth")
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # 新增：记录验证集最高正确率                                             #####55555555
        tb_writer.add_scalar("best_val_acc", best_val_acc, epoch)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")


    print(f"Training completed. Best validation accuracy: {best_val_acc}")
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds.')
    wandb.log({'total_training_time': total_time})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"F:\mmpre\deep-learning-for-image-processing\data\nutsv2ifolder")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符


    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
