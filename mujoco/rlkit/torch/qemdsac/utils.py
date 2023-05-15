import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
from scipy.stats import norm

class LinearSchedule(object):

    def __init__(self, schedule_timesteps, initial=1., final=0.):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final = final
        self.initial = initial

    def __call__(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial + fraction * (self.final - self.initial)


    
    
    

def calculate_iqem(tau,V):
    V_inverse = V.inverse()
   
    tau = tau.squeeze().cpu().numpy()
    batch_size,num = tau.shape[0],tau.shape[1]
    
    normal_quantile = torch.FloatTensor(np.array(norm(0,1).ppf(tau))).to(ptu.device) #(256,32)
    qem_2 = normal_quantile.unsqueeze(2)
    qem_1 = ptu.ones(batch_size,num).unsqueeze(2)
    qem_3 = (qem_2.pow(2)-1)
    qem_4 = (qem_2.pow(3)-3*qem_2)
    
    qem = torch.cat([qem_1,qem_2,qem_3,qem_4],dim=2) #(256,32,4)
    
    a = torch.bmm(qem.transpose(1,2),torch.matmul(V_inverse,qem)).inverse()
    b = torch.bmm(a,qem.transpose(1,2))
    c = torch.matmul(b,V_inverse)
            
    return c    

    
    
    



def calculate_qem(tau,V):
    V_inverse = V.inverse()
    tau = tau[0]
    tau = tau.squeeze().cpu().numpy()
    num = tau.shape[0] #32
    normal_quantile = ptu.zeros(num)
    
    qem_1 = ptu.ones(num)
    qem_2 = ptu.ones(num)
    qem_3 = ptu.ones(num)
    qem_4 = ptu.ones(num)
    
    for i in range(num):
        normal_quantile[i]=torch.FloatTensor(np.array(norm(0,1).ppf(tau[i])))
        qem_2[i] = normal_quantile[i]
        qem_3[i] = normal_quantile[i].pow(2)-1
        qem_4[i] = normal_quantile[i].pow(3)-3*normal_quantile[i]
        
    qem_t = torch.cat([qem_1.unsqueeze(0),qem_2.unsqueeze(0),qem_3.unsqueeze(0),qem_4.unsqueeze(0)],dim=0)
    qem = qem_t.T
    a = torch.mm(qem_t.mm(V_inverse.mm(qem)).inverse(),qem_t)
    b = torch.mm(a,V_inverse)
    return b