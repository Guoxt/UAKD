import os
import numpy as np
import time
import torch
from torch import nn, optim, distributions
from model import UNet, uakd_dual
from dataset import MyData
from torch.utils.data import DataLoader, CWKLLoss, kl_divergence_pixelwise
from tqdm import tqdm
from torch.autograd import Variable
from sets import *

def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    val_meter=AverageMeter()
    val_losses, dcs = [], []
    #criterion = t.nn.CrossEntropyLoss()
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input.cuda())
        val_label = Variable(label.cuda())
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
            model = model.cuda()
        outputs=model(val_input)
        pred = outputs.data.max(1)[1].cpu().numpy().squeeze()
        gt = val_label.data.cpu().numpy().squeeze()

        for i in range(gt.shape[0]):
            #print(i)
            dc,val_loss=calc_dice(gt[i,:,:,:],pred[i,:,:,:])
            dcs.append(dc)
            val_losses.append(val_loss)
        #for gt_, pred_ in zip(gt, pred):
            #gts.append(gt_)
            #preds.append(pred_)
    #score,cc,acc=scores(gts,preds,n_class=classes)
    model.train()
    return np.mean(dcs),np.mean(val_losses)

def run():

    # train
    lr = 0.0001
    batch_size = 10
    
    model = uakd_dual(in_ch=1, out_ch=1) 
    Smodel = U_Net(in_ch=1, out_ch=2) 
    Tmodel = U_Net(in_ch=1, out_ch=2) 
    
    if opt.use_gpu: 
        model.cuda()
        model.train()
        Smodel.cuda()
        Smodel.train()
        Tmodel.cuda()

    check_memory = ['...,...']

    train_data=MyData(opt.train_data_root,train=True)
    val_data=MyData(opt.train_data_root,train=False,val=True)
    val_dataloader = DataLoader(val_data,4,shuffle=False,num_workers=opt.num_workers)

    criterionn = t.nn.CrossEntropyLoss()#weight=weight
    criterion = CWKLLoss()
    
    if opt.use_gpu: 
        criterion = criterion.cuda()
        criterionn = criterionn.cuda()

    loss_meter=AverageMeter()
    previous_loss = 1e+20

    train_dataloader = DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=opt.num_workers)
    optimizer = t.optim.Adam(list(model.parameters()) + list(Smodel.parameters()),lr = lr,weight_decay = opt.weight_decay)

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()

        for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)):

            input = Variable(data)
            target = Variable(label)

            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            # Uncertainty generation
            Tdata = []
            for index,checkname in enumerate(check_memory):
                
                Tmodel.load_state_dict(t.load(checkname))
                Tmodel.eval()
                with torch.no_grad():
                    Tscore = Tmodel(input)
                prob = torch.nn.Softmax(dim=1)(Tscore).squeeze().detach().cpu().numpy()       
                Tdata.append(prob.transpose(1,0,2,3))

            softmax_predictions = np.array(Tdata)
            # calculate variance
            variance = np.var(softmax_predictions, axis=0)
            # calculate entropy
            entropy = -np.sum(softmax_predictions * np.log(softmax_predictions + 1e-10), axis=0)  # 加入一个小量避免log(0)问题
            # Setting a very small lower bound value
            eps = 1e-10
            # Calculate the mean softmax output
            avg_softmax = np.mean(softmax_predictions, axis=0)
            # Calculate the per-pixel KL divergence of each set of softmax outputs with respect to the mean
            kl_divs_pixelwise = []
            for softmax_outputs in softmax_predictions:
                kl_divs = kl_divergence_pixelwise(softmax_outputs, avg_softmax)
                kl_divs_pixelwise.append(kl_divs)
            kl_divs_pixelwise = np.array(kl_divs_pixelwise)
            # Calculate the per-pixel mutual information
            mutual_information = np.mean(kl_divs_pixelwise, axis=0)

            Pre = np.argmax((avg_softmax).astype(float),axis=0)
            Conf = avg_softmax.transpose(1,0,2,3)
            U1 = (variance[:,:,:,:]).transpose(1,0,2,3)
            U2 = (entropy[:,:,:,:]).transpose(1,0,2,3)
            U3 = (mutual_information).transpose(1,0,2,3)        

            # Train uakd model
            INPUT = torch.from_numpy(np.concatenate((np.concatenate((Conf[:,0:1,:,:],1-U1[:,0:1,:,:]), axis=1),np.concatenate((Conf[:,1:2,:,:],1-U1[:,1:2,:,:]), axis=1)), axis=0)).float()
            with torch.no_grad():
                INPUT = Variable(INPUT)
                #TARGET = Variable(TARGET)
            if opt.use_gpu:
                INPUT = INPUT.cuda()
                #TARGET = TARGET.cuda() 
            weights = model(INPUT)
            WEIGHTS = torch.cat((weights[0:weights.shape[0]//2,:,:,:], weights[weights.shape[0]//2:weights.shape[0],:,:,:]), dim=1)

            T = 2.5
            alpha = 0.99
            Tprob = F.softmax(Tscore/T, dim = 1)
            Sscore = Smodel(input)
            Sprob = F.softmax(Sscore, dim = 1)
            loss = alpha*criterion(Tprob, Sprob, WEIGHTS) + (1-alpha)*criterionn(Sscore, target)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())

            print('train-loss-avg:', loss_meter.avg,'train-loss-each:', loss_meter.val
    
        acc = 0
        val_loss = 0
        prefix = str(acc)+'_'+str(val_loss) + '_'+str(0)+'_'+str(batch_size)+'_'
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(model.state_dict(), name)
            
        name1 = time.strftime('%m%d_%H:%M:%S.npy')
        np.save(name1, plt_list)
    
if __name__ == '__main__':

    parser = get_default_experiment_parser()
    parser.add_argument("-p", "--patch_size", type=int, nargs="+", default=112)
    parser.add_argument("-in", "--in_channels", type=int, default=4)
    parser.add_argument("-lt", "--T", type=int, default=3)
    parser.add_argument("-lb", "--labels", type=int, nargs="+", default=[0, 1, 2, 3])
    args, _ = parser.parse_known_args()
    DEFAULTS, MODS = make_defaults(patch_size=args.patch_size, in_channels=args.in_channels, latent_size=args.latent_size, labels=args.labels)
    
    run()
