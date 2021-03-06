import argparse
import logging
import torch.cuda

from torch.utils.data import DataLoader
from iam_data_loader.iam_loader import IAMLoader
from generated_data_loader.generated_loader import GeneratedLoader
from models.crnn import CRNN

try:
    from .config import *
    from .reading_discriminator import *
except:
    from config import *
    from reading_discriminator import *






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
# - experiment arguments
#parser.add_argument('--load_model', '-lm', action='store', default=None,
#                    help='The name of the pretrained model to load. Defalt: None, i.e. random initialization')
#parser.add_argument('--save_model', '-sm', action='store', default='whateva.pt',
#                    help='The name of the file to save the model')


args = parser.parse_args()

#cudnn.benchmark = True

# print out the used arguments
logger.info('###########################################')
logger.info('Experiment Parameters:')
for key, value in vars(args).items():
    logger.info('%s: %s', str(key), str(value))
logger.info('###########################################')

# prepare datset loader

logger.info('Loading dataset')

if args.data_set == 'IAM':
    train_set = IAMLoader('train', level=data_name, fixed_size=(128, None))
    test_set = IAMLoader('test', level=data_name, fixed_size=(128, None))
elif args.data_set == 'Generated':
    train_set = GeneratedLoader(set= 'train', nr_of_channels=1, fixed_size=(128, None))
    test_set = GeneratedLoader(set= 'test', nr_of_channels=1, fixed_size=(128, None))
# augmentation using data sampler
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)


# load CNN
logger.info('Preparing Net')
net = CRNN(1, len(classes), 256)
#net = HTRNet(cnn_cfg, rnn_cfg, len(classes))#




loss = torch.nn.CTCLoss()
net_parameters = net.parameters()
nlr = args.learning_rate
optimizer = torch.optim.Adam(net_parameters, nlr, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5 * max_epochs), int(.75 * max_epochs)])

logger.info('Initializing Reading Discriminator')
rd = ReadingDiscriminator(optimizer, net, loss, 1e-4, load_model_name)


logger.info('Training:')
for epoch in range(1, max_epochs + 1):

    rd.train_on_Dataloader(epoch, train_loader, test_set, scheduler)

    if epoch % 1 == 0:
        logger.info('Saving net after %d epochs', epoch)
        rd.saveModel(save_model_name)

