import numpy as np
from abc import ABC, abstractmethod
import random

# ==============================================================
# Traveler
# ==============================================================

# ==============================================================
# Base class for shared attributes and methods
# ==============================================================
class TravelerBase(ABC):
    """Abstract base for traveler entities."""
    def __init__(self, traveler_type: int):
        self.traveler_type = traveler_type

    @abstractmethod
    def update_policy(self, system: 'System'):
        pass

# ==============================================================
# TravelerGroup (shared type-level parameters and learning)
# ==============================================================
class TravelerGroup(TravelerBase):
    def __init__(self, type_id: int, phi: np.ndarray, t_star: int, u_value: np.ndarray, K: int, T: int, delta: float=0.9, eta: float=0.1, alpha: float=42, beta: float=42, gamma: float=42):
        '''
        TODO: describe all the variables
        '''
        # Fixed attributes
        super().__init__(type_id)
        self.phi = np.array(phi)
        self.t_star = t_star
        self.u_value = np.array(u_value)
        self.delta = delta
        self.eta = eta  
        self.U = len(self.u_state) # MM: check that len(array) works
        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Dynamic attributes
        self.travelers: list[Traveler] = []
        # Shared policy structures (group-level learning)
        self.zeta = np.zeros((self.U * (self.K+1) * self.T * (self.K+1))) 
        self.V = np.zeros((self.U * (self.K+1)))
        self.Q = np.zeros((self.U * (self.K+1) * self.T * (self.K+1)))
        self.pi = np.random.rand(self.U * (self.K+1), self.T * (self.K+1))
        for i in range(self.U * (self.K+1)):
            k = i % (self.K+1)
            self.pi[i, k+1:] = 0
        self.pi = self.pi / self.pi.sum(axis=1, keepdims=True)

    def register(self, traveler: 'Traveler'):
        self.travelers.append(traveler)

    def update_policy(self, system: 'System'):
        # Computing the expected total reward from state (u,k) taking action (t,b)
        self.immediate_reward(system)
        self.Q = self.zeta + self.delta * np.dot(self.P, self.V) # MM: self.P to be defined
        
        # update policy
        new_pi = self.policiy_logit()
        self.pi = (1 - self.eta) * self.pi + self.eta * new_pi

        # update value
        Q_reshape = self.Q.reshape(self.U*(self.K+1), self.T*(self.K+1))
        pi_reshape = self.pi.reshape(self.U*(self.K+1), self.T*(self.K+1))
        self.V = np.sum(Q_reshape * pi_reshape, axis=1)
        return
    
    def immediate_reward(self, system: 'System'):
        # compute zeta
        for u in range(self.U):
          for k in range(self.K+1):
            for t in range(self.T):
              for b in range(self.K+1):
                idx = u * (self.K+1) * self.T * (self.K+1) + k * self.T * (self.K+1) + t * (self.K+1) + b # MM : check with KZ
                if b > system.b_star[t]:
                  if t <= self.t_star:
                    self.zeta[idx] = -self.u_value[u] * np.abs(t - self.t_star) * self.beta # fast lane + early
                  else:
                    self.zeta[idx] = -self.u_value[u] * np.abs(t - self.t_star) * self.gamma # fast lane + late
                elif b == system.b_star[t]:
                  if t <= self.t_star:
                    zeta_fast_lane = -self.u_value[u] * np.abs(t - self.t_star) * self.beta
                    zeta_slow_lane = -self.u_value[u] * (np.abs(t - self.t_star) * self.beta + system.slow_lane_queue[t] * self.alpha)
                    self.zeta[idx] = zeta_fast_lane * system.psi[t] + zeta_slow_lane * (1-system.psi[t]) # limit between fast/slow lane + early 
                  else:
                    zeta_fast_lane = -self.u_value[u] * np.abs(t - self.t_star) * self.gamma
                    zeta_slow_lane = -self.u_value[u] * (np.abs(t - self.t_star) * self.gamma + system.slow_lane_queue[t] * self.alpha)
                    self.zeta[idx] = zeta_fast_lane * system.psi[t] + zeta_slow_lane * (1-system.psi[t]) # limit between fast/slow lane + late
                else:
                  if t <= self.t_star:
                    self.zeta[idx] = -self.u_value[u] * (np.abs(t - self.t_star) * self.beta + system.slow_lane_queue[t] * self.alpha) # slow lane + early
                  else:
                    self.zeta[idx] = -self.u_value[u] * (np.abs(t - self.t_star) * self.gamma + system.slow_lane_queue[t] * self.alpha) # slow lane + late
        return 

    def policiy_logit(self):
        Q_mtx = self.Q.reshape(self.pi.shape)
        pi = np.exp(Q_mtx)/np.sum(np.exp(Q_mtx), axis=1).reshape(-1,1)
        return pi
    
    def __repr__(self):
        return f"TravelerGroup(type={self.traveler_type}, n={len(self.travelers)})"
    
# ==============================================================
# Inidividual Traveler (agent-level state and actions)
# ==============================================================
class Traveler(TravelerBase):
    def __init__(self, 
                 group: TravelerGroup,
                 k_init: int, 
                 delta_t: int,
                 id: int
                 ):
        '''
        TODO: describe all the variables
        '''
        super().__init__(group.traveler_type)
        # Group level
        self.group = group
        group.register(self)

        # Fixed attributes
        self.id = id
        self.time_slots = [i * delta_t for i in range(self.group.T)]

        # Dynamic attributes
        self.u_curr = 0
        self.k_curr = k_init 
        self.t = self.t_star
        self.b = 0
        self.use_fast_lane: bool = False

    def action(self):
        idx_row = self.u_curr * (self.group.K+1) + self.k_curr
        prob = self.group.pi[idx_row]
        idx_col = np.random.choice(self.group.T * (self.group.K+1), p=prob)
        self.t = idx_col / (self.group.K+1)
        self.b = idx_col % (self.group.K+1)

        # MM: reset fast lane usage, better place ? -> not the good place
        self.use_fast_lane = False 
        return 

    def paid_karma_bid(self):
        """Deduct the bid from the traveler's karma balance."""
        if self.use_fast_lane:
            self.k_curr -= self.b
        return
    
    def get_new_karma(self, karma):
        """
        Receive redistributed karma from the system.
        (See System.karma_redistribution())
        """
        self.k_curr += karma
        return
    
    def update_urgency(self):
        """Update urgency level based on transition matrix phi."""
        self.u_curr = np.random.choice(self.group.U, p=self.group.phi[self.u_curr])
        return     

# ==============================================================
# System
# ==============================================================

class System: 
    def __init__(self, fast_lane_capacity: int, slow_lane_capacity: int, K: int, T: int, travelers: list[Traveler]):
        '''
        TODO: describe all the variables
        '''
        # Fixed attributes
        self.fast_lane_capacity = fast_lane_capacity
        self.slow_lane_capacity = slow_lane_capacity
        self.K = K  
        self.T = T

        # Dynamic attributes
        self.b_star = np.zeros(self.T)            
        self.slow_lane_queue = np.zeros(self.T)    
        self.psi = np.zeros(self.T)        
        self.travelers = travelers

    def karma_redistribution(self):
        """
        Uniform redistribution of used karma among all travelers.
        TODO: 
        - Later : redistribute based on traveler states.
        """
        total_karma_used = sum(t.b for t in self.travelers if t.use_fast_lane)
        karma_per_traveler = total_karma_used // len(self.travelers)
        remainder = total_karma_used % len(self.travelers)  # leftover karma
        indexes_with_extra = set(random.sample(range(len(self.travelers)), remainder))
        for i, traveler in enumerate(self.travelers):
            extra = 1 if i in indexes_with_extra else 0
            traveler.get_new_karma(karma_per_traveler + extra)
        return

    def cost_lane(self, group_travelers: list[list[Traveler]]):
        """
        Loop over each departure time group and compute:
        - who enters the fast vs slow lane,
        - how the queue evolves in the slow lane.
        """
        # Reset dynamic attributes
        self.b_star = np.zeros(self.T)            
        self.slow_lane_queue = np.zeros(self.T)    
        self.psi = np.zeros(self.T) 
        # for each departure time
        for t, travelers in enumerate(group_travelers): 
            if len(travelers) == 0:
                self.b_star[t] = 0
                self.psi[t] = 1
                self.update_slow_lane_queue_length(
                    t, 0
                )
            else:
                self.b_star[t], self.psi[t]  = self.compute_b_star(travelers)
                for traveler in travelers:
                    if traveler.b > self.b_star[t]:
                        traveler.use_fast_lane = True
                    elif traveler.b == self.b_star[t]:
                        traveler.use_fast_lane = ... # KZ: sample based on self.psi[t]
                self.update_slow_lane_queue_length(
                    t, sum(1 for traveler in travelers if not traveler.use_fast_lane)
                )
        return 
    
    def compute_b_star(self, travelers: list[Traveler]):
        """
        Compute threshold bid (b_star) for the fast lane.

        """
        bids = sorted([traveler.b for traveler in travelers], reverse=True)
        if len(bids) > self.fast_lane_capacity:
            b_star = bids[self.fast_lane_capacity - 1]
            psi = ... # KZ: get probability of getting into fast lane if someone bids b_star
        else:
            b_star = bids[-1]
            psi = 1
        return b_star, psi

    def update_slow_lane_queue_length(self, t:int, inflow: int):
        """
        Simple bottleneck model for the slow lane.

        - If inflow (n_slow_lane) > capacity: queue builds up.
        - If inflow < capacity: queue dissipates gradually.
        """
        capacity = self.slow_lane_capacity
        if t == 0:
          self.slow_lane_queue[t] = max(0, 0 - (capacity - inflow))
        else:
          self.slow_lane_queue[t] = max(0, self.slow_lane_queue[t-1] - (capacity - inflow))
        return