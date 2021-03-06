#import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class gatedPool_l(nn.Module):
    def __init__(self, kernel_size, stride, padding = 0, dilation=1):
        super(gatedPool_l, self).__init__()
        self.mask = nn.Parameter(torch.randn(1,1,kernel_size,kernel_size)) # nn.Parameter is special Variable
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = False
        self.ceil_mode = False
    def forward(self, x):
        size = list(x.size())[1]
        out_size = list(x.size())[2] // 2
        bs = list(x.size())[0]
        xc = []
        for c in range(size):
            a = x[:,c,:,:]
            a = torch.unsqueeze(a,1)
            a = F.conv2d(a,self.mask,stride = self.stride)
            xc.append(a)
        output = xc[0]
        #print(output)
        for xx in xc[1:]:
            output = torch.cat((output,xx),1)
        #output = torch.reshape(xc,(bs,size,out_size,out_size))
        
        alpha = F.sigmoid(output)
    
        x = alpha*F.max_pool2d(x,self.kernel_size,self.stride,self.padding, self.dilation) + (1-alpha)*F.avg_pool2d(x,self.kernel_size,self.stride,self.padding)
        
        return x 

class gatedPool_c(nn.Module):
    def __init__(self,in_channel, kernel_size, stride, padding = 0, dilation=1):
        super(gatedPool_c, self).__init__()
        out_channel = in_channel
        self.mask = nn.Parameter(torch.randn(in_channel,out_channel,kernel_size,kernel_size)) # nn.Parameter is special Variable
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
         
    def forward(self, x):
        size = list(x.size())[1]
        out_size = list(x.size())[2] // 2
        bs = list(x.size())[0]
        mask_c = F.conv2d(x,self.mask,stride = self.stride)
        alpha = F.sigmoid(mask_c)
        #print(alpha)
        x = alpha*F.max_pool2d(x,self.kernel_size,self.stride,self.padding, self.dilation) + (1-alpha)*F.avg_pool2d(x,self.kernel_size,self.stride, self.padding)
        return x 
class mixedPool(nn.Module):
    def __init__(self, alpha, kernel_size, stride, padding = 0, dilation=1):
        # nn.Module.__init__(self)
        super(mixedPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha) # nn.Parameter is special Variable
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
         
    def forward(self, x):
        x = self.alpha*F.max_pool2d(x,self.kernel_size,self.stride,self.padding, self.dilation) +(1-self.alpha)*F.avg_pool2d(x,self.kernel_size,self.stride, self.padding)
        #print(self.alpha)
        #print(x.size())
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # mixed pooling 
        #self.mixedPool1 = mixedPool(alpha = 0.5,kernel_size=2,stride=2,padding = 0)
        #getedpooling
        self.mixedPool1 = gatedPool_c(in_channel=10,kernel_size=2,stride=2,padding = 0)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.mixedPool2 = mixedPool(alpha = 0.5,kernel_size=2,stride=2,padding = 0)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        x = self.mixedPool1(x)
        #print(x.size())
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.mixedPool2(x)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
