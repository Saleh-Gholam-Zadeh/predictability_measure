import torch
import numpy as np
from matplotlib import pyplot as plt
import random
import torch.nn.functional  as F

nn = torch.nn
import time
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rmse( predicted , target):
    """Mean Squared Error"""
    assert target.size() == predicted.size(), "shape mismatch // target.shape = " + str(target.size()) + "and predicted_mean.shape =" + str(predicted.size()) + "are provided"
    return torch.sqrt(torch.mean((target - predicted)**2))

def mse( predicted, target):
    """Mean Squared Error"""
    assert target.size() == predicted.size(), "shape mismatch // target.shape = " + str(target.size()) + "and predicted_mean.shape =" + str(predicted.size()) + "are provided"
    mseloss = nn.MSELoss()
    return mseloss(predicted, target)

    return torch.mean((target - predicted)**2)



def mae( predicted ,target):
    """Mean Absolute Error"""
    assert target.size() == predicted.size(), "shape mismatch // target.shape = " + str(target.size()) + "and predicted_mean.shape =" + str(predicted.size()) + "are provided"
    return torch.mean(torch.abs(target - predicted))

def kl_rmse( predicted_logits, target,lambda1=0.5,lambda2=0.5 ,normalization=None ,should_normalize=False):
    """Mean Absolute Error"""
    assert target.size() == predicted_logits.size(), "shape mismatch // target.shape = " + str(target.size()) + "and predicted_mean.shape =" + str(predicted_logits.size()) + "are provided"

    # KL(P||Q)
    kl_loss = nn.KLDivLoss(reduction="sum")
    pred_kl = F.log_softmax(predicted_logits[:,:,:-1],dim=-1) #predicted logits to probability distribution bins
    target_kl = target[:,:,:-1]
    #print("target_kl[0,10:14,:]:",target_kl[0,10:14,:]) ---> sometimes it is all zeros ---> therefore not necessarily a probability distribution
    dist_error = kl_loss(pred_kl, target_kl)/(pred_kl.shape[0]*pred_kl.shape[1])

    #MSE(packets)
    if(normalization is not None and should_normalize):
        packet_error = mse(predicted_logits[:,:,-1:],target[:,:,-1:])/torch.from_numpy(normalization['packets'][1]).to(device)**2
    else:
        packet_error = mse(predicted_logits[:, :, -1:], target[:, :, -1:])

    return lambda1*dist_error + lambda2*packet_error , dist_error , packet_error

def kl_loss(predicted_logits, target,manual=True):
    assert target.size() == predicted_logits.size(), "shape mismatch // target.shape = " + str(target.size()) + "and predicted_mean.shape =" + str(predicted_logits.size()) + "are provided"
    # KL(P||Q)
    kl_loss = nn.KLDivLoss(reduction="sum")
    pred_kl = F.log_softmax(predicted_logits[:, :, :-1], dim=-1)
    target_kl = target[:, :, :-1]
    dist_error = kl_loss(pred_kl, target_kl)/(pred_kl.shape[0]*pred_kl.shape[1])
    return dist_error



def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


values1 = [1.346112,1.337432,1.246655]
values2 = [1.033836,1.082015,1.117323]

def gaussian_nll(target, predicted_mean, predicted_var):
    """Gaussian Negative Log Likelihood (assuming diagonal covariance)"""
    assert target.size() == predicted_mean.size(), "shape mismatch // target.shape = " + str(target.size()) + "and predicted_mean.shape = " + str(predicted_mean.size()) + "are provided"
    predicted_var += 1e-12
    mahal = (target - predicted_mean)**2 / predicted_var
    element_wise_nll = 0.5 * (torch.log(predicted_var) + np.log(2 * np.pi) + mahal)
    sample_wise_error = torch.sum(element_wise_nll, dim=-1)
    return torch.mean(sample_wise_error)

def root_mean_squared(pred, target, data=[], tar='observations', fromStep=0, denorma=False, plot=None):
    """
    root mean squared error
    :param target: ground truth positions
    :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
    :return: root mean squared error between targets and predicted mean, predicted variance is ignored
    """
    pred = pred[..., :target.shape[-1]]

    sumSquare = 0
    count = 0
    if plot != None:
        for idx in range(target.shape[2]):
            plt.plot(target[3,:,idx],label='target')
            plt.plot(pred[3,:,idx],label='prediction')
            plt.legend()
            plt.show()

    # if denorma==True:
    #     pred = denorm(pred, data, tar)
    #     target = denorm(target, data, tar)



    #target = target[:, fromStep:, :]
   # pred = pred[:, fromStep:, :]
    numSamples = 1
    for dim in target.shape:
        numSamples = numSamples * dim
    #print('RMSE Samplesss......................................',numSamples)
    sumSquare = np.sum(np.sum(np.sum((target - pred) ** 2)))
    return np.sqrt(sumSquare / numSamples)



def one_hotting(tens,max_class):

    '''
    tens: torch tensor input of shape (a_1,a_2,...,a_lastdim)
    output: torch tensor of  shape (a1,a2,...,a_lastdim,max(tens+1))
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("one_hotting is called...")
    #mx= torch.max(tens)
    one_hot = (torch.arange(max_class).cpu().detach().numpy() == tens[..., None].cpu().detach().numpy()).astype(int) #give cuda error once using gpu! not necessary to solve
    one_hot = torch.from_numpy(one_hot)
    return (one_hot).to(device) #checked works perfectly!


def CrossEntropy(pred,tch_gt,C=20,manual=False):
    '''
    pred: output of Neural Net whcih is a torch Tensor. the last dim refers to the score of each class  (still its not probability) [N,d1,...,dk,C]. C=number of classes. # however it should be transformed to [N,C , d1, d2 , ... ,dk]
    K CAN BE ZERO and we can have [N,C] and [N] as prediction and groundtruth
    tch_gt:[N,d1,...,dk] # before 1hot encoding. groundtruth in torch tensor format
    C=totl number of classes(dimension of output when NN is applied to input feature-vector)
    manual:bool  True=manual-implementation   False=torch-implementation
    '''

    assert len(tch_gt.shape) == 3  and len(tch_gt.shape) == 3
    B,T_pred,class_pred = pred.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #assert tch_gt.size()[0] == pred.size()[0],"N1 == N2 is not hold true:"+str(tch_gt.size()) + " and " +str(pred.size()) +"are provided!"

    # one_hot_gt = one_hotting(tch_gt, max_class=C)
    # one_hot_gt.cpu().detach().numpy()
    if (manual==False):
        #print("pytorch cross-Entropy loss out of the box....")
        loss = nn.CrossEntropyLoss(reduction='mean')
        pred_for_pytorch_loss = pred.transpose(1,-1)  # size = [N,C,d1,..,dk]. in my case [N,d1,C]. C is in the last dim (at the start) and should be put in 2nd dim to be used in Loss

        idx = 1  # [N,C,d1,d2,...] index of C
        sz1_to_be_checked = pred_for_pytorch_loss.size()[:idx] + pred_for_pytorch_loss.size()[idx + 1:]
        assert sz1_to_be_checked == tch_gt.size()[:-1], "wrong size for pytorch cross entropy loss.  Pred should be [N,C,d1,..,dk]. here we check N,di in both input but it seems that" + str(
            sz1_to_be_checked) + " and " + str(tch_gt.size()) + " are provided which are not matched"
        assert pred_for_pytorch_loss.size()[idx] == C, "length of 1Hot should match with number of classes but" + str(pred_for_pytorch_loss.size()[idx]) + " and " + str(C) + "are provided"
        tch_gt = tch_gt.type(torch.LongTensor)
        tch_gt = (tch_gt).to(device)
        pred_for_pytorch_loss = (pred_for_pytorch_loss).to(device)
        return loss(pred_for_pytorch_loss, tch_gt[:,:,0])
    elif(manual==True):
        print("manual calculations.....")
        one_hot_gt = one_hotting(tch_gt, max_class=C)
        one_hot_gt.cpu().detach().numpy()

        probs = nn.Softmax(dim=-1)(pred)
        #print("probs = ", probs, "log(probs) = ", torch.log(probs), "one_hot_gt = ", one_hot_gt)
        # torch.log(probs)

        #print(" torch.mul(one_hot_gt,torch.log(probs)) = ", torch.mul(one_hot_gt, torch.log(probs)))
        li = -torch.sum(torch.mul(one_hot_gt, torch.log(probs)))
        normalization_factor = np.prod(probs.size()) / probs.size()[-1]  # or pred_for_pytorch_loss.size() [1]. multiplication of all sizes except C. N*d1*...*dk

        return li / normalization_factor

if __name__ == '__main__':

# example to convert integer label to one-hot  and how to convert back
#     labels = torch.randint(0, 19, (2,5))
#     print(labels.size())
#     print(labels)
#     # labels --> one-hot
#     one_hot = torch.nn.functional.one_hot(labels, num_classes=20)
#     print("one_hot.size()",one_hot.size())
#     print("one_hot = ",one_hot)
#     # one-hot --> labels
#     labels_again = torch.argmax(one_hot, dim=-1)
#     print("labels_again.shape = ",labels_again.shape)
#     print("labels_again = ",labels_again)
    print("--------------------------------1st example----------------------")
    print("checking torch implementation & manual implementation of CrossEntropyLoss")

    #Generating a sample pred and gt....
    #possible range of output class is between r1,r2:
    r1=0 #min_pred
    r2=19 #max_pred
    output_dim=20  #finally make it 100 for packet classification
    pred = torch.FloatTensor(2,5,output_dim).uniform_(r1, r2) #asample prediction of arbitrary size
    print('pred = ',pred) # pred is score of each class. still it is not a prob because it is unnormalized
    print("pred.size() = ",pred.size())
    #gt_size = tuple(pred.size()) +(out_put_dim,) #[10,75,1]
    gt_size = pred.size()[:-1]  #[10,75]
    #print("gt_size= ",gt_size)
    tch_gt = torch.randint(r1,r2,gt_size) #0to99 (min,possible_max-1,(shape))
    print("tch_gt = ",tch_gt)
    print("tch_gt.size() = " ,tch_gt.size())

    s1 = time.time()
    print(CrossEntropy(pred, tch_gt, C=output_dim, manual=False))
    s2 = time.time()
    print(CrossEntropy(pred, tch_gt, C=output_dim, manual=True)) #torch implementation is 1.5-2 times faster than manual
    s3 = time.time()

    print((s2-s1)/(s3-s2))