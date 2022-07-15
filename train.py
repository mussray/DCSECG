import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from models.rkccsnet import *
from models.csnet import *
from loss import *
import torch.optim as optim
from data_processor import *
from trainer import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    global args
    args = parser.parse_args()
    setup_seed(1)

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        torch.backends.cudnn.benchmark = True

    if args.model == 'rkccsnet':
        model = CSNet1(sensing_rate=args.sensing_rate)
    elif args.model == 'csnet':
        model = CSNet(sensing_rate=args.sensing_rate)


    model = model.cuda()
    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90, 120, 150, 180], gamma=0.25, last_epoch=-1)
    train_loader, valid_loader = data_loader()
    print('train_loader')
    print (train_loader.shape)
    print('valid_loader')
    print (valid_loader.shape)

    print('\nModel: %s\n'
          'Sensing Rate: %.2f\n'
          'Epoch: %d\n'
          'Initial LR: %f\n'
          % (args.model, args.sensing_rate, args.epochs, args.lr))

    print('Start training')
    doc = open("./save_temp/result.txt", 'w+')
    for epoch in range(args.epochs):
        print('\ncurrent lr {:.5e}'.format(optimizer.param_groups[0]['lr']),file=doc)
        loss = train(train_loader, model, criterion, optimizer, epoch,doc)
        scheduler.step()
        if epoch%5==0 or epoch==199:
            psnr,prd= valid(valid_loader, model, criterion)
            print("Total Loss: %f" % loss,file=doc)
            print("Total Loss: %f" % loss)
            print("PSNR: %f" % psnr,file=doc)
            print("PSNR: %f" % psnr)
            print("PRD: %f" % prd,file=doc)
            print("PRD: %f" % prd)
        print("Total Loss: %f" % loss)
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'mine'+str(epoch+1)+'.pth'))
    print('Trained finished.')
    print('Model saved in %s' % (os.path.join(args.save_dir, args.model+'.pth')))

    input0=valid_loader[0]
    input1 = torch.from_numpy(
        (input0.numpy() - np.min(input0.numpy())) / (np.max(input0.numpy()) - np.min(input0.numpy())))#.cuda()

    output0 = model(input1)

    input0=valid_loader[0].numpy().reshape(256)

    plt.plot(range(256),input0,'r',range(256),output0[0].cpu().detach().reshape(256)*(np.max(input0) - np.min(input0)) + np.min(input0),'b')
    plt.savefig('./save_temp/result.png', dpi=200)
    plt.show()

'''
    for inputs, _ in enumerate(valid_loader):
        inputs = rgb_to_ycbcr(inputs)[:, 0, :, :].unsqueeze(1) / 255.
        outputs = model(inputs)
        plt.imshow(inputs)
        plt.imshow(outputs)
'''



if __name__ == '__main__':
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rkccsnet',
                        choices=['csnet', 'rkccsnet'],
                        help='choose model to train')
    #defult=0.50000
    parser.add_argument('--sensing-rate', type=float, default=0.05,
                        choices=[0.50000, 0.25000, 0.12500, 0.06250, 0.03125],
                        help='set sensing rate')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--block-size', default=256, type=int,
                        metavar='N', help='block size (default: 32)')
    parser.add_argument('--image-size', default=50, type=int,
                        metavar='N', help='image size used for training (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4 , type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='pth', type=str)

    main()












