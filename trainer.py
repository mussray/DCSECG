from utils import *
from matplotlib import pyplot as plt
import sys
import argparse
import torch
ber_matrix=np.random.choice([-1,1],(25,256))/np.sqrt(25)
ber_matrix=np.random.randn(25,256)
#model1=torch.load('.\\pth\\model_199.pth')
def prdcal(inputs,outputs):
    n,c,x=inputs.shape
    sum1=0
    sum2=0
    for i in range(x):
            sum1=sum1+(inputs[0,0,i]-outputs[0,0,i])*(inputs[0,0,i]-outputs[0,0,i])
            sum2=sum2+(inputs[0,0,i])*(inputs[0,0,i])
    prd=np.sqrt((sum1/sum2).cpu())*100
    snr=10*log10(sum2/sum1)

    return prd,snr
def train(train_loader, model, criterion, optimizer, epoch,doc):
    print('Epoch: %d' % (epoch + 1),file=doc)
    print('mine-5 Epoch: %d' % (epoch + 1))
    model.train()
    sum_loss = 0
    t=0
    for inputs in train_loader:#1,1,256
        original=inputs
        #inputs = rgb_to_ycbcr(inputs)[:, 0, :, :].unsqueeze(1) / 255.
        inputs = torch.from_numpy((inputs.numpy() - np.min(inputs.numpy())) / (np.max(inputs.numpy())-np.min(inputs.numpy()))).cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        #loss = criterion(outputs[0], inputs) + criterion(outputs[1], inputs) + criterion(outputs[2], inputs)
        loss = criterion(outputs[0], inputs)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        t=t+1
        if (t%10000==0):
            print(str(t)+'/48000')


    return sum_loss/t


def valid(valid_loader, model, criterion):
    sum_snr = 0
    sum_ssim = 0
    sum_prd = 0
    _ssim = SSIM().cuda()
    model.eval()
    with torch.no_grad():
        s=0
        for iters, (inputs) in enumerate(valid_loader):
            original=inputs
            #inputs = rgb_to_ycbcr(inputs)[:, 0, :, :].unsqueeze(1) / 255.
            inputs = torch.from_numpy((inputs.numpy() - np.min(inputs.numpy())) / (np.max(inputs.numpy()) - np.min(inputs.numpy()))).cuda()
            outputs = model(inputs)
            outputs=outputs[0]*(np.max(original.numpy()) - np.min(original.numpy())) + np.min(original.numpy())
            prd,snr=prdcal(original,outputs)
            sum_prd += prd
            sum_snr += snr
            s=s+1
            if (s % 4000 == 0):
                print(str(round(s/12000*100,2))+'%')

    return sum_snr / (iters), sum_prd/(iters)

'''
def test(test_loader, model, criterion):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM()
    model.test()
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(test_loader):
            inputs = rgb_to_ycbcr(inputs)[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(outputs[0], inputs)
    return sum_psnr / (iters), sum_ssim / (iters)
'''

