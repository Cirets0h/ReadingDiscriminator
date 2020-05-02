import argparse
import logging

import numpy as np
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import tqdm
import torch.backends.cudnn as cudnn

from iam_data_loader.iam_loader import IAMLoader
from generated_data_loader.generated_loader import GeneratedLoader

from utils.auxilary_functions import image_resize
try:
    from .config import *
except:
    from config import *



from models.htr_net import HTRNet
from models.crnn import CRNN

from utils.save_load import my_torch_save, my_torch_load

from utils.auxilary_functions import torch_augm

from os.path import isfile

import torch.nn.functional as F

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('PHOCNet-Experiment::train')
logger.info('--- Running PHOCNet Training ---')
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                    help='lr')
parser.add_argument('--data_set', '-ds', choices=['IAM', 'Generated'], default='IAM',
                    help='Which dataset is used for training')
parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                    help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
parser.add_argument('--display', action='store', type=int, default=50,
                    help='The number of iterations after which to display the loss values. Default: 100')
parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
# - experiment arguments
#parser.add_argument('--load_model', '-lm', action='store', default=None,
#                    help='The name of the pretrained model to load. Defalt: None, i.e. random initialization')
#parser.add_argument('--save_model', '-sm', action='store', default='whateva.pt',
#                    help='The name of the file to save the model')


args = parser.parse_args()

# train as:
# -lrs 5000:1e-4,10000:1e-5 -bs 1 -is 10 -fim 36  -gpu 0 --test_interval 1000

gpu_id = args.gpu_id
#cudnn.benchmark = True


device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

# print out the used arguments
logger.info('###########################################')
logger.info('Experiment Parameters:')
for key, value in vars(args).items():
    logger.info('%s: %s', str(key), str(value))
logger.info('###########################################')

# prepare datset loader

logger.info('Loading dataset.')

if args.data_set == 'IAM':
    train_set = IAMLoader('train', level=data_name, fixed_size=(128, None))
    test_set = IAMLoader('test', level=data_name, fixed_size=(128, None))
elif args.data_set == 'Generated':
    train_set = GeneratedLoader(nr_of_channels=1, fixed_size=(128, None))
    test_set = GeneratedLoader(nr_of_channels=1, fixed_size=(128, None))
# augmentation using data sampler
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)


# load CNN
logger.info('Preparing Net...')
net = CRNN(1, len(classes), 256)
#net = HTRNet(cnn_cfg, rnn_cfg, len(classes))#

if load_model_name is not None:
    my_torch_load(net, load_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.device_count())
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    net = nn.DataParallel(net)
net.to(device)


loss = torch.nn.CTCLoss()
net_parameters = net.parameters()
nlr = args.learning_rate
optimizer = torch.optim.Adam(net_parameters, nlr, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5 * max_epochs), int(.75 * max_epochs)])

from PIL import Image
import os

image = np.array(Image.open(os.path.join('2.png')).convert('1'))
image = image[:, :, np.newaxis]
image = torch.Tensor(image).float().unsqueeze(0)
image = image.transpose(1, 3).transpose(2, 3)
#
tst_o = net(Variable(image.cuda()))
tdec = tst_o.log_softmax(2).argmax(2).permute(1, 0).cpu().numpy().squeeze()
tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
estimated_word = ''.join([icdict[t] for t in tt]).replace('_', '')
print('greedy dec: ' + estimated_word)
def train(epoch):
    optimizer.zero_grad()
    pos = 0
    pos_5 = 0
    closs = []
    for iter_idx, (img, transcr) in enumerate(train_loader):

        img = Variable(img.to(device))
        # cuda augm - alternatively for cpu use it on dataloader
        img = torch_augm(img)
        output = net(img)

        act_lens = torch.IntTensor(img.size(0)*[output.size(0)])
        labels = Variable(torch.IntTensor([cdict[c] for c in ''.join(transcr)]))
        label_lens = torch.IntTensor([len(t) for t in transcr])

        output = output.log_softmax(2) #.detach().requires_grad_()

        loss_val = loss(output.cpu(), labels, act_lens, label_lens)
        closs += [loss_val.data]

        loss_val.backward()

        if iter_idx % iter_size == iter_size - 1:
            optimizer.step()
            optimizer.zero_grad()


        if iter_idx % 50 == 1:
            logger.info('Epoch %d, Iteration %d: %f', epoch, iter_idx+1, sum(closs)/len(closs))
            #tt = [v for j, v in enumerate(output) if j == 0 or v != tdec[j - 1]]
            print(loss_val.data)
            #print(tt)
            closs = []
            try:
                tst_img, tst_transcr = test_set.__getitem__(np.random.randint(test_set.__len__()))
                with torch.no_grad():
                    tst_o = net(Variable(tst_img.cuda()).unsqueeze(0))
                tdec = tst_o.log_softmax(2).argmax(2).permute(1, 0).cpu().numpy().squeeze()
                #for i, tdec in enumerate(declbls):

                print('orig:: ' + tst_transcr)
                  # todo: create a better way than to just ignore output with size [1, 1, 80] (first 1 has to be >1
                tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            except:
                tt = ''
                print('Error occured')
            estimated_word = ''.join([icdict[t] for t in tt]).replace('_', '')
            if estimated_word == tst_transcr:
                pos += 1
                if len(estimated_word) > 5:
                    pos_5 += 1

            print('greedy dec: ' + estimated_word)
            print('Accuracy: ' + str(pos) + '/' + str(int(iter_idx/50+1)) + '= ' +str(pos/int(iter_idx/50+1)) + '| over 5 chars: ' + str(pos_5))
        #tdec, _, _, tdec_len = decoder.decode(tst_o.softmax(2).permute(1, 0, 2))
        #print('beam dec:: ' + ''.join([icdict[t.item()] for t in tdec[0, 0][:tdec_len[0, 0].item()]]))

import editdistance
# slow implementation
def test(epoch):
    net.eval()

    logger.info('Testing at epoch %d', epoch)
    cer, wer = [], []
    for (img, transcr) in test_loader:
        transcr = transcr[0]
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            o = net(img)
        tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
        #tdec, _, _, tdec_len = decoder.decode(o.softmax(2).permute(1, 0, 2))
        #dec_transcr = ''.join([icdict[t.item()] for t in tdec[0, 0][:tdec_len[0, 0].item()]])

        cer += [float(editdistance.eval(dec_transcr, transcr))/ len(transcr)]
        wer += [float(editdistance.eval(dec_transcr.split(' '), transcr.split(' '))) / len(transcr.split(' '))]

    logger.info('CER at epoch %d: %f', epoch, sum(cer) / len(cer))
    logger.info('WER at epoch %d: %f', epoch, sum(wer) / len(wer))


    net.train()


cnt = 0
logger.info('Training:')
for epoch in range(1, max_epochs + 1):

    scheduler.step()
    train(epoch)


    if epoch % 1 == 0:
        logger.info('Saving net after %d epochs', epoch)
        my_torch_save(net, save_model_name)
        net.cuda(gpu_id)


#my_torch_save(net, save_model_name)