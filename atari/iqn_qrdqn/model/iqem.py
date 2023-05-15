import torch

from .base_model import BaseModel
from iqn_qrdqn.network import DQNBase, CosineEmbeddingNetwork,\
    QuantileNetwork
from iqn_qrdqn.utils import calculate_iqem

class IQEM(BaseModel):

    def __init__(self, num_channels, num_actions, K=32, num_cosines=32,
                 embedding_dim=7*7*64, dueling_net=False, noisy_net=False,):
        super(IQEM, self).__init__()

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels)
        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, embedding_dim=embedding_dim,
            noisy_net=noisy_net)
        # Quantile network.
        self.quantile_net = QuantileNetwork(
            num_actions=num_actions, dueling_net=dueling_net,
            noisy_net=noisy_net)

        self.K = K
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
      

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        # Sample fractions.
        taus = torch.rand(
            batch_size, self.K, dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        
        
        qem_weight = calculate_iqem(taus).to(state_embeddings.device) #(batch_size, 4, self.K)
        
        # Calculate quantiles.
        quantiles = self.calculate_quantiles(
            taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)
        
    
        
        # Calculate expectations of value distributions.
        q = torch.bmm(qem_weight,quantiles)[:,:2,:] #(batch_size, 1, self.num_actions)
        
        
        
        #q = quantiles.mean(dim=1)
        
        assert q.shape == (batch_size, 2,self.num_actions)

        return q
