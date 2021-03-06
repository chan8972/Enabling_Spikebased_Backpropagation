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
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

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
parser.add_argument('-b', '--batch-size', default=16, type=int,
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
parser.add_argument('--lr', '--learning-rate', default=0.0033, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')

best_prec1 = 0
change = 70
change2 = 100
change3 = 125

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

    print(model)
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(44)

    if device_num < 2:
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

    # Data loading code
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = torchvision.datasets.CIFAR10('./data_CIFAR10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_data = torchvision.datasets.CIFAR10('./data_CIFAR10', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_data,  # val_data for testing
                                             batch_size=int(args.batch_size/2), shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False)

    if args.evaluate:
        validate(val_loader, model, criterion, criterion_en, time_steps=100, leak=0.99)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        ep.append(epoch)

        # train for one epoch
        train(train_loader, model, criterion, criterion_en, optimizer, epoch, time_steps=100, leak=0.99)

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
              'Prec@5 {top5:.3f} \t'
              'En_Loss_Eval {losses_en_eval: .4f} \t'
              'Prec@1_tr {top1_tr:.3f} \t'
              'Prec@5_tr {top5_tr:.3f} \t'
              'En_Loss_train {losses_en: .4f}'.format(
            ep[k], args.epochs, lRate[k], top1=tp1[k], top5=tp5[k], losses_en_eval=losses_eval[k], top1_tr=tp1_tr[k],
            top5_tr=tp5_tr[k], losses_en=losses_tr[k]))


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
    # print ('mark1',train_loader.sampler)
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

        output, Total_11_output, LF_11_output, Total_12_output, LF_12_output, Total_21_output, LF_21_output, Total_22_output, LF_22_output, Total_31_output, LF_31_output, Total_32_output, LF_32_output, Total_33_output, LF_33_output, Total_f0_output, LF_f0_output, out11_temp, out12_temp, out21_temp, out22_temp, out31_temp, out32_temp, out33_temp, outf0_temp = model(input_var, steps=time_steps, l=leak)

        # compute gradient
        NG_C11 = grad_cal(leak, LF_11_output, Total_11_output)
        NG_C12 = grad_cal(leak, LF_12_output, Total_12_output)
        NG_C21 = grad_cal(leak, LF_21_output, Total_21_output)
        NG_C22 = grad_cal(leak, LF_22_output, Total_22_output)
        NG_C31 = grad_cal(leak, LF_31_output, Total_31_output)
        NG_C32 = grad_cal(leak, LF_32_output, Total_32_output)
        NG_C33 = grad_cal(leak, LF_33_output, Total_33_output)
        NG_F0 = grad_cal(leak, LF_f0_output, Total_f0_output)

        # apply gradient
        for z in range(time_steps):
            out11_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C11))
            out12_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C12))
            out21_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C21))
            out22_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C22))
            out31_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C31))
            out32_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C32))
            out33_temp[z].register_hook(lambda grad: torch.mul(grad, NG_C33))
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

        out11_temp, NG_C11 = None, None
        out12_temp, NG_C12 = None, None
        out21_temp, NG_C21 = None, None
        out22_temp, NG_C22 = None, None
        out31_temp, NG_C31 = None, None
        out32_temp, NG_C32 = None, None
        out33_temp, NG_C33 = None, None
        outf0_temp, NG_F0 = None, None

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}] Prec@1 {top1_tr.avg:.3f} Prec@5 {top5_tr.avg:.3f} Entropy_Loss {loss_en.avg:.4f}'
          .format(epoch, top1_tr=top1_tr, top5_tr=top5_tr, loss_en=losses_en))

    losses_tr.append(losses_en.avg)
    tp1_tr.append(top1_tr.avg)
    tp5_tr.append(top5_tr.avg)


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


def save_checkpoint(state, is_best, filename='checkpointT1_cifar10_v9.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_bestT1_cifar10_v9.pth.tar')


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
    lr = args.lr * (1 ** (epoch // change))

    for param_group in optimizer.param_groups:
        if epoch >= change3:
            param_group['lr'] = 0.2 * 0.2 * 0.2 * lr

        elif epoch >= change2:
            param_group['lr'] = 0.2 * 0.2 * lr

        elif epoch >= change:
            param_group['lr'] = 0.2 * lr

        else:
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
    membrane_potential = membrane_potential - ex_membrane # hard reset
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
    membrane_potential = membrane_potential - ex_membrane
    # generate spike
    out = SpikingNN()(ex_membrane)

    return membrane_potential, out


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.cnn21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.cnn31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2)

        self.fc0 = nn.Linear(256 * 4 * 4, 1024, bias=False)
        self.fc1 = nn.Linear(1024, 10, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2.0 / n)
                m.weight.data.normal_(0, variance1)
                # define threshold
                m.threshold = 1

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]  # number of columns
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)
                # define threshold
                m.threshold = 1

    def forward(self, input, steps=100, l=0.99, l2=0.99, DO=0.2, DO_f=0.2):
        drop = nn.Dropout(p=DO, inplace=True)
        drop_f = nn.Dropout(p=DO_f, inplace=True)

        out11_temp = []
        out12_temp = []
        out21_temp = []
        out22_temp = []
        out31_temp = []
        out32_temp = []
        out33_temp = []
        outf0_temp = []

        mem_11 = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda(), requires_grad=False)
        mask_11 = Variable(torch.ones(input.size(0), 64, 32, 32).cuda(), requires_grad=False)
        mask_11 = drop(mask_11)
        mem_12 = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda(), requires_grad=False)
        mask_12 = Variable(torch.ones(input.size(0), 64, 32, 32).cuda(), requires_grad=False)
        mask_12 = drop(mask_12)
        mem_1s = Variable(torch.zeros(input.size(0), 64, 16, 16).cuda(), requires_grad=False)

        mem_21 = Variable(torch.zeros(input.size(0), 128, 16, 16).cuda(), requires_grad=False)
        mask_21 = Variable(torch.ones(input.size(0), 128, 16, 16).cuda(), requires_grad=False)
        mask_21 = drop(mask_21)
        mem_22 = Variable(torch.zeros(input.size(0), 128, 16, 16).cuda(), requires_grad=False)
        mask_22 = Variable(torch.ones(input.size(0), 128, 16, 16).cuda(), requires_grad=False)
        mask_22 = drop(mask_22)
        mem_2s = Variable(torch.zeros(input.size(0), 128, 8, 8).cuda(), requires_grad=False)

        mem_31 = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        mask_31 = Variable(torch.ones(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        mask_31 = drop(mask_31)
        mem_32 = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        mask_32 = Variable(torch.ones(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        mask_32 = drop(mask_32)
        mem_33 = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        mask_33 = Variable(torch.ones(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        mask_33 = drop(mask_33)
        mem_3s = Variable(torch.zeros(input.size(0), 256, 4, 4).cuda(), requires_grad=False)

        membrane_f0 = Variable(torch.zeros(input.size(0), 1024).cuda(), requires_grad=False)
        mask_f0 = Variable(torch.ones(input.size(0), 1024).cuda(), requires_grad=False)
        mask_f0 = drop_f(mask_f0)

        membrane_f1 = Variable(torch.zeros(input.size(0), 10).cuda(), requires_grad=True)

        Total_11_output = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda(), requires_grad=False)
        LF_11_output = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda(), requires_grad=False)

        Total_12_output = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda(), requires_grad=False)
        LF_12_output = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda(), requires_grad=False)

        Total_21_output = Variable(torch.zeros(input.size(0), 128, 16, 16).cuda(), requires_grad=False)
        LF_21_output = Variable(torch.zeros(input.size(0), 128, 16, 16).cuda(), requires_grad=False)

        Total_22_output = Variable(torch.zeros(input.size(0), 128, 16, 16).cuda(), requires_grad=False)
        LF_22_output = Variable(torch.zeros(input.size(0), 128, 16, 16).cuda(), requires_grad=False)

        Total_31_output = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        LF_31_output = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)

        Total_32_output = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        LF_32_output = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)

        Total_33_output = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)
        LF_33_output = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False)

        Total_f0_output = Variable(torch.zeros(input.size(0), 1024).cuda(), requires_grad=False)
        LF_f0_output = Variable(torch.zeros(input.size(0), 1024).cuda(), requires_grad=False)

        for i in range(steps):
            # Poisson input spike generation
            rand_num = Variable(torch.rand(input.size(0), input.size(1), input.size(2), input.size(3)).cuda())
            Poisson_d_input = (torch.abs(input) > rand_num).type(torch.cuda.FloatTensor)
            Poisson_d_input = torch.mul(Poisson_d_input, torch.sign(input))

            # convolutional Layer
            mem_11 = mem_11 + self.cnn11(Poisson_d_input)
            mem_11, out = LIF_sNeuron(mem_11, self.cnn11.threshold, l, i)
            out = torch.mul(out, mask_11)
            LF_11_output, Total_11_output, out = LF_Unit(l, LF_11_output, Total_11_output, out, out11_temp, i)

            mem_12 = mem_12 + self.cnn12(out)
            mem_12, out = LIF_sNeuron(mem_12, self.cnn12.threshold, l, i)
            out = torch.mul(out, mask_12)
            LF_12_output, Total_12_output, out = LF_Unit(l, LF_12_output, Total_12_output, out, out12_temp, i)

            # pooling Layer
            mem_1s = mem_1s + self.avgpool1(out)
            mem_1s, out = Pooling_sNeuron(mem_1s, 0.75, i)

            # convolutional Layer
            mem_21 = mem_21 + self.cnn21(out)
            mem_21, out = LIF_sNeuron(mem_21, self.cnn21.threshold, l, i)
            out = torch.mul(out, mask_21)
            LF_21_output, Total_21_output, out = LF_Unit(l, LF_21_output, Total_21_output, out, out21_temp, i)

            mem_22 = mem_22 + self.cnn22(out)
            mem_22, out = LIF_sNeuron(mem_22, self.cnn22.threshold, l, i)
            out = torch.mul(out, mask_22)
            LF_22_output, Total_22_output, out = LF_Unit(l, LF_22_output, Total_22_output, out, out22_temp, i)

            # pooling Layer
            mem_2s = mem_2s + self.avgpool2(out)
            mem_2s, out = Pooling_sNeuron(mem_2s, 0.75, i)

            # convolutional Layer
            mem_31 = mem_31 + self.cnn31(out)
            mem_31, out = LIF_sNeuron(mem_31, self.cnn31.threshold, l, i)
            out = torch.mul(out, mask_31)
            LF_31_output, Total_31_output, out = LF_Unit(l, LF_31_output, Total_31_output, out, out31_temp, i)

            mem_32 = mem_32 + self.cnn32(out)
            mem_32, out = LIF_sNeuron(mem_32, self.cnn32.threshold, l, i)
            out = torch.mul(out, mask_32)
            LF_32_output, Total_32_output, out = LF_Unit(l, LF_32_output, Total_32_output, out, out32_temp, i)

            mem_33 = mem_33 + self.cnn33(out)
            mem_33, out = LIF_sNeuron(mem_33, self.cnn33.threshold, l, i)
            out = torch.mul(out, mask_33)
            LF_33_output, Total_33_output, out = LF_Unit(l, LF_33_output, Total_33_output, out, out33_temp, i)

            # pooling Layer
            mem_3s = mem_3s + self.avgpool3(out)
            mem_3s, out = Pooling_sNeuron(mem_3s, 0.75, i)

            out = out.view(out.size(0), -1)

            # fully-connected Layer
            membrane_f0 = membrane_f0 + self.fc0(out)
            membrane_f0, out = LIF_sNeuron(membrane_f0, self.fc0.threshold, l, i)
            out = torch.mul(out, mask_f0)
            LF_f0_output, Total_f0_output, out = LF_Unit(l, LF_f0_output, Total_f0_output, out, outf0_temp, i)

            membrane_f1 = membrane_f1 + self.fc1(out)
            membrane_f1 = l * membrane_f1.detach() + membrane_f1 - membrane_f1.detach()

        return membrane_f1/self.fc1.threshold/steps, Total_11_output, LF_11_output, Total_12_output, LF_12_output, Total_21_output, LF_21_output, Total_22_output, LF_22_output, Total_31_output, LF_31_output, Total_32_output, LF_32_output, Total_33_output, LF_33_output, Total_f0_output, LF_f0_output, out11_temp,out12_temp,out21_temp,out22_temp,out31_temp,out32_temp,out33_temp,outf0_temp


    def tst(self, input, steps=100, l=0.99):
        mem_11 = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda(), requires_grad=False, volatile=True)
        mem_12 = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda(), requires_grad=False, volatile=True)
        mem_1s = Variable(torch.zeros(input.size(0), 64, 16, 16).cuda(), requires_grad=False, volatile=True)

        mem_21 = Variable(torch.zeros(input.size(0), 128, 16, 16).cuda(), requires_grad=False, volatile=True)
        mem_22 = Variable(torch.zeros(input.size(0), 128, 16, 16).cuda(), requires_grad=False, volatile=True)
        mem_2s = Variable(torch.zeros(input.size(0), 128, 8, 8).cuda(), requires_grad=False, volatile=True)

        mem_31 = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False, volatile=True)
        mem_32 = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False, volatile=True)
        mem_33 = Variable(torch.zeros(input.size(0), 256, 8, 8).cuda(), requires_grad=False, volatile=True)
        mem_3s = Variable(torch.zeros(input.size(0), 256, 4, 4).cuda(), requires_grad=False, volatile=True)

        membrane_f0 = Variable(torch.zeros(input.size(0), 1024).cuda(), requires_grad=False, volatile=True)
        membrane_f1 = Variable(torch.zeros(input.size(0), 10).cuda(), requires_grad=False, volatile=True)

        for i in range(steps):
            # Poisson input spike generation
            rand_num = Variable(torch.rand(input.size(0), input.size(1), input.size(2), input.size(3)).cuda())
            Poisson_d_input = (torch.abs(input) > rand_num).type(torch.cuda.FloatTensor)
            Poisson_d_input = torch.mul(Poisson_d_input, torch.sign(input))

            # convolutional Layer
            mem_11 = mem_11 + self.cnn11(Poisson_d_input)
            mem_11, out = LIF_sNeuron(mem_11, self.cnn11.threshold, l, i)

            mem_12 = mem_12 + self.cnn12(out)
            mem_12, out = LIF_sNeuron(mem_12, self.cnn12.threshold, l, i)

            # pooling Layer
            mem_1s = mem_1s + self.avgpool1(out)
            mem_1s, out = Pooling_sNeuron(mem_1s, 0.75, i)

            # convolutional Layer
            mem_21 = mem_21 + self.cnn21(out)
            mem_21, out = LIF_sNeuron(mem_21, self.cnn21.threshold, l, i)

            mem_22 = mem_22 + self.cnn22(out)
            mem_22, out = LIF_sNeuron(mem_22, self.cnn22.threshold, l, i)

            # pooling Layer
            mem_2s = mem_2s + self.avgpool2(out)
            mem_2s, out = Pooling_sNeuron(mem_2s, 0.75, i)

            # convolutional Layer
            mem_31 = mem_31 + self.cnn31(out)
            mem_31, out = LIF_sNeuron(mem_31, self.cnn31.threshold, l, i)

            mem_32 = mem_32 + self.cnn32(out)
            mem_32, out = LIF_sNeuron(mem_32, self.cnn32.threshold, l, i)

            mem_33 = mem_33 + self.cnn33(out)
            mem_33, out = LIF_sNeuron(mem_33, self.cnn33.threshold, l, i)

            # pooling Layer
            mem_3s = mem_3s + self.avgpool3(out)
            mem_3s, out = Pooling_sNeuron(mem_3s, 0.75, i)

            out = out.view(out.size(0), -1)

            # fully-connected Layer
            membrane_f0 = membrane_f0 + self.fc0(out)
            membrane_f0, out = LIF_sNeuron(membrane_f0, self.fc0.threshold, l, i)

            membrane_f1 = membrane_f1 + self.fc1(out)
            membrane_f1 = l * membrane_f1.detach() + membrane_f1 - membrane_f1.detach()

        return membrane_f1 / self.fc1.threshold / steps

if __name__ == '__main__':
    main()