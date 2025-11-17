import numpy as np
from abc import ABC, abstractmethod

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
    def __init__(self, type_id: int, phi: np.ndarray, t_star: int, u_low: float, u_high: float, K: int, T: int, delta: float=0.9, eta: float=0.1):
        # KZ: urgency levels and transition kernels shall be input args
        
        super().__init__(type_id)
        self.phi = np.array(phi)
        self.t_star = t_star
        self.u_low = u_low
        self.u_high = u_high
        self.delta = delta
        self.eta = eta  

        # Traveler VOT 
        self.alpha = ...
        self.beta = ...
        self.gamma = ...

        # Travelers belonging to this group
        self.travelers: list[Traveler] = []

        # Shared policy structures (group-level learning)
        U = 2  # urgency levels
        self.zeta = np.zeros((U * (K+1) * T * (K+1))) 
        self.V = np.zeros((U * (K+1)))
        self.Q = np.zeros((U * (K+1) * T * (K+1)))
        self.pi = np.random.rand(U * (K+1), T * (K+1))
        for i in range(U * (K+1)):
            k = i % (K+1)
            self.pi[i, k+1:] = 0
        self.pi = self.pi / self.pi.sum(axis=1, keepdims=True)

    def register(self, traveler: 'Traveler'):
        self.travelers.append(traveler)

    def update_policy(self, system: 'System'):
        """TODO: Define policy update logic.""" 
        U = ...
        K = ...
        T = ...
        self.immediate_reward(system)
        self.Q = self.zeta + self.delta * np.dot(self.P, self.V)
        
        # update policy
        new_pi = self.policiy_logit()
        self.pi = (1 - self.eta) * self.pi + self.eta * new_pi

        # update value
        Q_reshape = self.Q.reshape(U*(K+1), T*(K+1))
        pi_reshape = self.pi.reshape(U*(K+1), T*(K+1))
        self.V = np.sum(Q_reshape * pi_reshape, axis=1)
        return
    
    def immediate_reward(self, system: 'System'):
        """TODO: Compute immediate (negative) expected cost or utility.
        should be part of the update policy function -> could be computed in the system I think
        """
        # global index selection and value
        n = self.zeta.size
        u = np.empty(n)
        u[:n//2] = self.u[0]
        u[n//2:] = self.u[1]

        psi = system.psi # calculate in the cost lane function of the system
        K = 1000 # avoid magic numbers here

        time_= self.alpha * system.t_q + self.beta * system.t_e[:,:,self.traveler_type] + self.gamma * system.t_l[:,:,self.traveler_type]
        time = np.repeat(time, K+1, axis=0)

        self.zeta = np.dot(-u , (psi*time).sum(axis=1)) 
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
                 T: int,
                 K: int,
                 id: int
                 ):
        super().__init__(group.traveler_type)
        self.group = group
        group.register(self)

        # --- Individual state ---
        self.id = id
        self.k = k_init
        self.u_curr = 0
        self.t = self.t_star
        self.b = 0
        self.time_slots = [i * delta_t for i in range(T)]
        self.use_fast_lane: bool = False


    def action(self):
        """
        TODO: Define a better logic.
        
        """
        # KZ: sample from group.pi

        self.t = self.t_star  

        if self.u_curr == 1:
            self.b = self.k
        else:
            self.b = 0

        # MM: reset fast lane usage, better place ?
        self.use_fast_lane = False 
        return 

    def paid_karma_bid(self):
        """Deduct the bid from the traveler's karma balance."""
        if self.use_fast_lane:
            self.k -= self.b
        return
    
    def get_new_karma(self, karma):
        """
        Receive redistributed karma from the system.
        (See System.karma_redistribution())
        """
        self.k += karma
        return
    
    def update_urgency(self):
        """Update urgency level based on transition matrix phi."""
        self.u_curr = np.random.choice(len(self.u), p=self.phi[self.u_curr])
        return     

# ==============================================================
# System
# ==============================================================

class System: 
    def __init__(self, fast_lane_capacity: int, slow_lane_capacity: int, K: int, T: int, travelers: list[Traveler]):

        # --- Fixed attributes ---
        self.fast_lane_capacity = fast_lane_capacity
        self.slow_lane_capacity = slow_lane_capacity

        # --- Change  ---
        self.b_star = None            # threshold bid for fast lane
        self.slow_lane_queue:int = 0     # current queue length in slow lane
        # dim2 for slow/fast lane and dim3 for traveler types
        # KZ: no need to defined early/late arrival but directly computed using t_star only once for each group/type
        self.t_q: np.ndarray = np.zeros(T)
        self.t_e: np.ndarray = np.zeros((T,2,3))
        self.t_l: np.ndarray = np.zeros((T,2,3))

        
        self.travelers = travelers

    def karma_redistribution(self):
        """
        TODO: 
        - First redistribute uniformly.
        - Later, redistribute based on traveler states.
        """
        # KZ: let first make uniform redistribution as default policy
        # an easier logit here:
        # preliminary: define binary attribute for each traveler "use_fast_lane"
        # step 1: collect and accumulate karma form all travelers, each based on self.use_fast_lane, self.b
        # step 2: compute karma for redistribution: total karma collected / number of travelers
        # step 3: redistribute karma using traveler method self.get_new_karma
        total_karma_used = sum(t.b for t in self.travelers if t.use_fast_lane)

        karma_per_traveler = total_karma_used // len(self.travelers)
        remainder = total_karma_used % len(self.travelers)  # leftover karma
        for i, traveler in enumerate(self.travelers):
            extra = 1 if i < remainder else 0
            traveler.get_new_karma(karma_per_traveler + extra)
        return

    def cost_lane(self, group_travelers: list[list[Traveler]]):
        """
        Loop over each departure time group and compute:
        - who enters the fast vs slow lane,
        - how the queue evolves in the slow lane.

        group_travelers: list of lists, 
        each sublist = travelers with the same departure time.
        """
        # KZ: this function is super messy now, let's do it together

        costs = []
        self.slow_lane_queue = 0  
        self.psi = np.zeros_like(self.psi) 
        for index, travelers in enumerate(group_travelers): # index represents time 
            if len(travelers) == 0:
                self.b_star = 0
                self.update_slow_lane_queue_length(
                    0
                )
            else:
                self.compute_b_star(travelers)
                for traveler in travelers:
                    if traveler.b >= self.b_star:
                        traveler.use_fast_lane = True
                self.update_slow_lane_queue_length(
                    sum(1 for t in travelers if not t.use_fast_lane)
                )
            
            K = 1000  # avoid magic numbers here
            
            # used to compute the Immediate Reward Function
            self.psi[index*(K+1)+self.b_star : (index+1)*(K+1),1] = 1 # not an attribute anymore -> find solution
            
            self.t_q[index,0] = self.slow_lane_queue/self.slow_lane_capacity # time spent in the queue
            for traveler in travelers:
                self.t_e[index,:, traveler.traveler_type] = max(0, traveler.t_star - traveler.t - self.t_q[index,:])
                self.t_l[index,:, traveler.traveler_type] = max(0, traveler.t + self.t_q[index,:] - traveler.t_star)

        # Who bid for the fast lane -> determine fast/slow lane assignment
        # Then determine how many people enter the slow lane
        # Update the queue length
        # Repeat these steps for every departure time
        return 
    
    def compute_b_star(self, travelers: list[Traveler]):
        """
        Compute threshold bid (b_star) for the fast lane.

        """
        bids = sorted([traveler.b for traveler in travelers], reverse=True)
        if len(bids) > self.fast_lane_capacity:
            self.b_star = bids[self.fast_lane_capacity - 1]
        else:
            self.b_star = bids[-1]
        return

    def update_slow_lane_queue_length(self, n_slow_lane: int):
        """
        Simple bottleneck model for the slow lane.

        - If inflow (n_slow_lane) > capacity: queue builds up.
        - If inflow < capacity: queue dissipates gradually.
        """
        inflow = n_slow_lane
        capacity = self.slow_lane_capacity
        self.slow_lane_queue = max(0, self.slow_lane_queue - (capacity - inflow))
        return 

    # based on t, b and same state, compute the cost 