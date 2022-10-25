import torch
import torch.nn as nn
import torch.optim
import os
import argparse
import dataloader
import model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    net = model.MyNet().cuda()
    net.apply(weights_init)
    if config.load_pretrain:
        net.load_state_dict(torch.load(config.pretrain_dir))

    train_dataset = dataloader.pair_image_loader(config.data_path, config.label_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    loss1 = nn.CrossEntropyLoss().cuda()

    net.train()
    loss_str = ''
    floss = open("loss.csv", "w")
    for epoch in range(config.num_epochs):
        for iteration, data in enumerate(train_loader):
            input = data['data'].cuda()
            label = data['label'].cuda()
            torch.autograd.set_detect_anomaly(True)
            out = net(input)
            bs, _, _, _ = label.shape
            x = out.reshape([1,bs])
            y = label.reshape([1,bs])

            loss = loss1(x, y)
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Total Loss at iteration", iteration + 1, ":", loss.item(), "\n")
                loss_str += str(loss.item())
                loss_str += ','
        torch.save(net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
    floss.write(loss_str)
    floss.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--data_path', type=str, default="./train_data/data/")
    parser.add_argument('--label_path', type=str, default='./train_data/label/')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots-1/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
