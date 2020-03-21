import argparse
import os
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as dsets

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST]')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-load', default='', type=str, metavar='PATH',
                    help='path to training mask (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--lr', '--learning-rate', default=0.0085, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=125, type=int, metavar='N',
                    help='number of total epochs to run')

best_prec1 = 0
change = 75
tp1 = [];
tp5 = [];
ep = [];
lRate = [];
device_num = 1

tp1_tr = [];
tp5_tr = [];
losses_tr = [];
losses_eval = [];

def main():
    global args, best_prec1, batch_size, device_num

    args = parser.parse_args()
    batch_size = args.batch_size
    model = CNNModel()
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(42)

    if device_num < 3:
        device = 0
        torch.cuda.set_device(device)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(size_average=False).cuda()
    criterion_en = torch.nn.CrossEntropyLoss()

    learning_rate = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    '''STEP 1: LOADING DATASET'''
    train_data = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    val_data = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=int(args.batch_size), shuffle=False)

    if args.evaluate:
        validate(val_loader, model, criterion, criterion_en, time_steps=100, leak=0.99)
        return

    prec1_tr = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        ep.append(epoch)

        # train for one epoch
        prec1_tr = train(train_loader, model, criterion, criterion_en, optimizer, epoch, time_steps=100, leak=0.99)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, criterion_en, time_steps=100, leak=0.99)


        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    for k in range(0, args.epochs - args.start_epoch):
        print('Epoch: [{0}/{1}]\t'
              'LR:{2}\t'
              'Prec@1 {top1:.3f} \t'
              'Prec@5 {top5:.3f} '.format(
            ep[k], args.epochs, lRate[k], top1=tp1[k], top5=tp5[k]))


def grad_cal(l, LF_output, Total_output):
    Total_output = Total_output + (Total_output < 1e-3).type(torch.cuda.FloatTensor)
    out = LF_output.gt(1e-3).type(torch.cuda.FloatTensor) + math.log(l) * torch.div(LF_output, Total_output)
    return out


def train(train_loader, model, criterion, criterion_en, optimizer, epoch, time_steps, leak):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_tr = AverageMeter()
    top5_tr = AverageMeter()
    losses_en = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        labels = Variable(target.cuda())
        if device_num < 2:
            input_var = Variable(input.cuda())
        else:
            input_var = torch.autograd.Variable(input)

        optimizer.zero_grad()  # Clear gradients w.r.t. parameters

        output, Total_1_output, LF_1_output, Total_2_output, LF_2_output, Total_f0_output, LF_f0_output, out1_temp, out2_temp, outf0_temp = model(input_var, steps=time_steps, l=leak)

        # compute gradient
        NG_C1 = grad_cal(leak, LF_1_output, Total_1_output)
        NG_C2 = grad_cal(leak, LF_2_output, Total_2_output)
        NG_F0 = grad_cal(leak, LF_f0_output, Total_f0_output)

        # apply gradient
        for z in range(time_steps):
            out1_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C1))
            out2_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C2))
            outf0_temp[z].register_hook(lambda grad: torch.mul(grad, NG_F0))

        targetN = output.data.clone().zero_().cuda()
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))

        loss = criterion(output, targetN)
        loss_en = criterion_en(output, labels).cuda()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        prec1_tr, prec5_tr = accuracy(output.data, target, topk=(1, 5))
        losses_en.update(loss_en.data[0], input.size(0))
        top1_tr.update(prec1_tr[0], input.size(0))
        top5_tr.update(prec5_tr[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward(retain_variables=False)
        optimizer.step()

        out1_temp, NG_C1 = None, None
        out2_temp, NG_C2 = None, None
        outf0_temp, NG_F0 = None, None

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}] Prec@1 {top1_tr.avg:.3f} Prec@5 {top5_tr.avg:.3f} Entropy_Loss {loss_en.avg:.4f}'
          .format(epoch, top1_tr=top1_tr, top5_tr=top5_tr, loss_en=losses_en))

    losses_tr.append(losses_en.avg)
    tp1_tr.append(top1_tr.avg)
    tp5_tr.append(top5_tr.avg)

    return top1_tr.avg


def validate(val_loader, model, criterion, criterion_en, time_steps, leak):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_en_eval = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        labels = Variable(target.cuda())
        target = target.cuda(async=True)
        if device_num < 2:
            input_var = Variable(input.cuda())
        else:
            input_var = torch.autograd.Variable(input)

        output = model.tst(input=input_var, steps=time_steps, l=leak)

        targetN = output.data.clone().zero_().cuda()
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))
        loss = criterion(output, targetN)
        loss_en = criterion_en(output, labels).cuda()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        losses_en_eval.update(loss_en.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Entropy_Loss {losses_en_eval.avg:.4f}'
          .format(top1=top1, top5=top5, losses_en_eval=losses_en_eval))

    tp1.append(top1.avg)
    tp5.append(top5.avg)
    losses_eval.append(losses_en_eval.avg)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpointT1_mnist1.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_bestT1_mnist1.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr

    for param_group in optimizer.param_groups:
        if epoch >= change:
            param_group['lr'] = 0.2 * lr

        elif epoch < change:
            param_group['lr'] = lr

    lRate.append(param_group['lr'])


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SpikingNN(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(0).type(torch.cuda.FloatTensor)

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0.0] = 0
        return grad_input


def LIF_sNeuron(membrane_potential, threshold, l, i):
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane
    # generate spike
    out = SpikingNN()(ex_membrane)
    # decay
    membrane_potential = l * membrane_potential.detach() + membrane_potential - membrane_potential.detach()
    out = out.detach() + torch.div(out, threshold) - torch.div(out, threshold).detach()

    return membrane_potential, out


def LF_Unit(l, LF_output, Total_output, out, out_temp, i):
    LF_output = l * LF_output + out
    Total_output = Total_output + out
    out_temp.append(out)

    return LF_output, Total_output, out_temp[i]


def Pooling_sNeuron(membrane_potential, threshold, i):
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike
    out = SpikingNN()(ex_membrane)

    return membrane_potential, out


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2, bias=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.fc0 = nn.Linear(50*7*7, 200, bias=False)
        self.fc1 = nn.Linear(200, 10, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2. / n)
                m.weight.data.normal_(0, variance1)
                # define threshold
                m.threshold = 1

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)
                # define threshold
                m.threshold = 1

    def forward(self, input, steps=100, l=0.99):
        out1_temp = []
        out2_temp = []
        outf0_temp = []

        mem_1 = Variable(torch.zeros(input.size(0), 20, 28, 28).cuda(), requires_grad=False)
        mem_1s = Variable(torch.zeros(input.size(0), 20, 14, 14).cuda(), requires_grad=False)

        mem_2 = Variable(torch.zeros(input.size(0), 50, 14, 14).cuda(), requires_grad=False)
        mem_2s = Variable(torch.zeros(input.size(0), 50, 7, 7).cuda(), requires_grad=False)

        membrane_f0 = Variable(torch.zeros(input.size(0), 200).cuda(), requires_grad=False)
        membrane_f1 = Variable(torch.zeros(input.size(0), 10).cuda(), requires_grad=True)

        Total_1_output = Variable(torch.zeros(input.size(0), 20, 28, 28).cuda(), requires_grad=False)
        LF_1_output = Variable(torch.zeros(input.size(0), 20, 28, 28).cuda(), requires_grad=False)

        Total_2_output = Variable(torch.zeros(input.size(0), 50, 14, 14).cuda(), requires_grad=False)
        LF_2_output = Variable(torch.zeros(input.size(0), 50, 14, 14).cuda(), requires_grad=False)

        Total_f0_output = Variable(torch.zeros(input.size(0), 200).cuda(), requires_grad=False)
        LF_f0_output = Variable(torch.zeros(input.size(0), 200).cuda(), requires_grad=False)

        for i in range(steps):
            # Poisson input spike generation
            rand_num = Variable(torch.rand(input.size(0), input.size(1), input.size(2), input.size(3)).cuda())
            Poisson_d_input = ((torch.abs(input)/2) > rand_num).type(torch.cuda.FloatTensor)
            Poisson_d_input = torch.mul(Poisson_d_input, torch.sign(input))

            # convolutional Layer
            mem_1 = mem_1 + self.cnn1(Poisson_d_input)
            mem_1, out = LIF_sNeuron(mem_1, self.cnn1.threshold, l, i)
            LF_1_output, Total_1_output, out = LF_Unit(l, LF_1_output, Total_1_output, out, out1_temp, i)

            # pooling Layer
            mem_1s = mem_1s + self.avgpool1(out)
            mem_1s, out = Pooling_sNeuron(mem_1s, 0.75, i)

            # convolutional Layer
            mem_2 = mem_2 + self.cnn2(out)
            mem_2, out = LIF_sNeuron(mem_2, self.cnn2.threshold, l, i)
            LF_2_output, Total_2_output, out = LF_Unit(l, LF_2_output, Total_2_output, out, out2_temp, i)

            # pooling Layer
            mem_2s = mem_2s + self.avgpool2(out)
            mem_2s, out = Pooling_sNeuron(mem_2s, 0.75, i)

            out = out.view(out.size(0), -1)

            # fully-connected Layer
            membrane_f0 = membrane_f0 + self.fc0(out)
            membrane_f0, out = LIF_sNeuron(membrane_f0, self.fc0.threshold, l, i)
            LF_f0_output, Total_f0_output, out = LF_Unit(l, LF_f0_output, Total_f0_output, out, outf0_temp, i)

            membrane_f1 = membrane_f1 + self.fc1(out)
            membrane_f1 = l * membrane_f1.detach() + membrane_f1 - membrane_f1.detach()

        return membrane_f1 /self.fc1.threshold / steps, Total_1_output, LF_1_output, Total_2_output, LF_2_output, Total_f0_output, LF_f0_output, out1_temp, out2_temp, outf0_temp


    def tst(self, input, steps=100, l=0.99):
        mem_1 = Variable(torch.zeros(input.size(0), 20, 28, 28).cuda(), requires_grad=False, volatile=True)
        mem_1s = Variable(torch.zeros(input.size(0), 20, 14, 14).cuda(), requires_grad=False, volatile=True)

        mem_2 = Variable(torch.zeros(input.size(0), 50, 14, 14).cuda(), requires_grad=False, volatile=True)
        mem_2s = Variable(torch.zeros(input.size(0), 50, 7, 7).cuda(), requires_grad=False, volatile=True)

        membrane_f0 = Variable(torch.zeros(input.size(0), 200).cuda(), requires_grad=False, volatile=True)
        membrane_f1 = Variable(torch.zeros(input.size(0), 10).cuda(), requires_grad=False, volatile=True)

        for i in range(steps):
            # Poisson input spike generation
            rand_num = Variable(torch.rand(input.size(0), input.size(1), input.size(2), input.size(3)).cuda())
            Poisson_d_input = ((torch.abs(input)/2) > rand_num).type(torch.cuda.FloatTensor)
            Poisson_d_input = torch.mul(Poisson_d_input, torch.sign(input))

            # convolutional Layer
            mem_1 = mem_1 + self.cnn1(Poisson_d_input)
            mem_1, out = LIF_sNeuron(mem_1, self.cnn1.threshold, l, i)

            # pooling Layer
            mem_1s = mem_1s + self.avgpool1(out)
            mem_1s, out = Pooling_sNeuron(mem_1s, 0.75, i)

            # convolutional Layer
            mem_2 = mem_2 + self.cnn2(out)
            mem_2, out = LIF_sNeuron(mem_2, self.cnn1.threshold, l, i)

            # pooling Layer
            mem_2s = mem_2s + self.avgpool2(out)
            mem_2s, out = Pooling_sNeuron(mem_2s, 0.75, i)

            out = out.view(out.size(0), -1)

            # fully-connected Layer
            membrane_f0 = membrane_f0 + self.fc0(out)
            membrane_f0, out = LIF_sNeuron(membrane_f0, self.fc0.threshold, l, i)

            membrane_f1 = membrane_f1 + self.fc1(out)
            membrane_f1 = l * membrane_f1.detach() + membrane_f1 - membrane_f1.detach()

        return membrane_f1 / self.fc1.threshold / steps

if __name__ == '__main__':
    main()
