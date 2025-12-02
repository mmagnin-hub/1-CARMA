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
        self.U = self.u_value.shape[0]
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
        ######################### VERIFICATION #############################################
        self.P = np.zeros((self.U * (self.K+1), self.U * (self.K+1)))
        for u_from in range(self.U):
            row_idx_start = u_from * (self.K+1) 
            row_idx_end = row_idx_start + (self.K+1)
            for u_to in range(self.U):
                col_idx_start = u_to * (self.K+1)
                col_idx_end = col_idx_start + (self.K+1)
                self.P[row_idx_start:row_idx_end, col_idx_start:col_idx_end] = self.phi[u_from, u_to] / (self.K+1)  
        self.simulated_P = np.zeros((self.U * (self.K+1), self.U * (self.K+1)))
        ###################################################################################

    def register(self, traveler: 'Traveler'):
        self.travelers.append(traveler)

    def update_transition_matrix(self):
        '''
        Update the transition matrix P based on simulated transitions.
        '''
        ############################# VERIFICATION ########################################
        for traveler in self.travelers:
            idx_state_init = traveler.u_start *(self.K + 1) + traveler.k_start 
            idx_state_final = traveler.u_curr *(self.K + 1) + traveler.k_curr
            self.simulated_P[idx_state_init,idx_state_final] += 1

        # update transition matrix
        for row in range(self.simulated_P.shape[0]):
            row_sum = self.simulated_P[row, :].sum()
            if row_sum > 0:
                self.simulated_P[row, :] /= row_sum
        self.P = 0.5* (self.P + self.simulated_P)
        self.simulated_P[:] = 0
        ###################################################################################
        return
    
    def update_policy(self, system: 'System'):
        # Computing the expected total reward from state (u,k) taking action (t,b)
        self.immediate_reward(system)
        V_reshape = np.repeat(self.V, self.T * (self.K + 1)).reshape(self.U*(self.K+1), self.T*(self.K+1)) # MM : check for the shape behavior
        self.Q = self.zeta + self.delta * np.dot(self.P, V_reshape).flatten()
        
        # update policy
        new_pi = self.policiy_logit()
        self.pi = (1 - self.eta) * self.pi + self.eta * new_pi # smoothing

        # update value
        Q_reshape = self.Q.reshape(self.U*(self.K+1), self.T*(self.K+1)) # MM : check  for the shape behavior
        self.V = np.sum(Q_reshape * self.pi, axis=1)
        return
    
    def immediate_reward(self, system: 'System'):
        '''
        TODO: add more matrix calculations to optimize
        '''
        # compute zeta
        for u in range(self.U):
          for k in range(self.K+1):
            for t in range(self.T):
              for b in range(self.K+1):
                idx = u * (self.K+1) * self.T * (self.K+1) + k * self.T * (self.K+1) + t * (self.K+1) + b 
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
        self.u_start = 0
        self.k_start = 0
        self.t: int = self.group.t_star
        self.b = 0
        self.use_fast_lane: bool = False

    def action(self):
        idx_row = self.u_curr * (self.group.K+1) + self.k_curr
        prob = self.group.pi[idx_row]
        idx_col = np.random.choice(self.group.T * (self.group.K+1), p=prob)
        self.t = idx_col // (self.group.K+1)
        self.b = idx_col % (self.group.K+1) # MM : how to prevent bidding more karma than current balance ? -> already in pi ? 
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

    def store_start_state(self):
        """ Store start of the day to compare at the end of the day. """
        self.u_start = self.u_curr
        self.k_start = self.k_curr  
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
        self.N = len(travelers)

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
        total_karma_used = sum(traveler.b for traveler in self.travelers if traveler.use_fast_lane)
        karma_per_traveler = total_karma_used // len(self.travelers)
        leftover_karma = total_karma_used % len(self.travelers) 
        indexes_with_extra = set(random.sample(range(len(self.travelers)), leftover_karma))
        for i, traveler in enumerate(self.travelers):
            extra = 1 if i in indexes_with_extra else 0
            traveler.get_new_karma(karma_per_traveler + extra)
        return

    def simulate_lane_queue(self):
        """
        Loop over each departure time group and compute:
        - who enters the fast vs slow lane,
        - how the queue evolves in the slow lane.
        """
        # Reset dynamic attributes
        self.b_star = np.zeros(self.T)            
        self.slow_lane_queue = np.zeros(self.T)    
        self.psi = np.zeros(self.T) 
        for traveler in self.travelers: traveler.use_fast_lane = False
        # sort travelers for each departure time
        group_travelers = self.group_travelers_by_departure() 
        for t, travelers in enumerate(group_travelers): 
            if len(travelers) != 0:
                self.b_star[t], self.psi[t]  = self.determine_threshold_bid(travelers)
                self.assign_lanes(t, travelers)
                count_slow_lane_users = sum(1 for traveler in travelers if not traveler.use_fast_lane)
                self.update_queue_slow_lane(t, count_slow_lane_users)
            else:
                self.b_star[t], self.psi[t] = 0, 1
                self.update_queue_slow_lane(t, 0)
        return 
    
    def group_travelers_by_departure(self):
        """
        Create a list of travelers for each departure time slot.
        """
        group_travelers = [[] for _ in range(self.T)]
        for traveler in self.travelers:
            group_travelers[traveler.t].append(traveler)
        return group_travelers

    def determine_threshold_bid(self, travelers: list[Traveler]):
        """
        Compute threshold bid (b_star) for the fast lane.
        """
        bids = sorted([traveler.b for traveler in travelers], reverse=True)
        if len(bids) > self.fast_lane_capacity:
            b_star = bids[self.fast_lane_capacity - 1]
            traveler_count_at_b_star = sum(1 for traveler in travelers if traveler.b == b_star)
            traveler_count_over_b_star = sum(1 for traveler in travelers if traveler.b > b_star)
            free_spot_at_b_star = self.fast_lane_capacity - traveler_count_over_b_star
            psi = free_spot_at_b_star/traveler_count_at_b_star 
        else:
            b_star = bids[-1]
            psi = 1
        return b_star, psi
    
    def assign_lanes(self, t, travelers):
        '''
        Update each traveler's lane choice (use_fast_lane) based on b_star and psi.
        '''
        for traveler in travelers:
            if traveler.b > self.b_star[t]:
                traveler.use_fast_lane = True
            elif traveler.b == self.b_star[t]:
                # MM: How to ensure that it doesn't create queue ?
                traveler.use_fast_lane = random.random() < self.psi[t] # MM: add a warning print
        return
    
    def update_queue_slow_lane(self, t:int, inflow: int):
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




