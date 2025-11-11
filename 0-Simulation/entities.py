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
    def __init__(self, type_id: int, phi: np.ndarray, t_star: int, u_low: float, u_high: float, K: int, T: int):
        super().__init__(type_id)
        self.phi = np.array(phi)
        self.t_star = t_star
        self.u_low = u_low
        self.u_high = u_high

        # Travelers belonging to this group
        self.travelers: list[Traveler] = []

        # Shared policy structures (group-level learning)
        U = 2  # urgency levels
        self.V = np.zeros((U, K+1))
        self.Q = np.zeros((U, K+1, T))
        self.pi = np.random.rand(U*(K+1), T)
        self.pi = self.pi / self.pi.sum(axis=1, keepdims=True)

    def register(self, traveler: 'Traveler'):
        self.travelers.append(traveler)

    def update_policy(self, system: 'System'):
        """TODO: Define policy update logic.""" 
        delta = 0.9
        self.immediate_reward(system)
        self.Q = self.zeta + delta * np.dot(self.P, self.V)
        return
    
    def immediate_reward(self, system: 'System'):
        """TODO: Compute immediate (negative) expected cost or utility.
        should be part of the update policy function -> could be computed in the system I think
        """
        ############# based on the paper KARMA
        # computed from the bottleneck outcome
        n = self.zeta.size

        u = np.empty(n)
        u[:n//2] = self.u[0]
        u[n//2:] = self.u[1]

        psi = system.psi # calculate in the cost lane function of the system
        K = 1000 # avoid magic numbers here
        alpha = ...
        beta = ...
        gamma = ...

        # for type 0
        time_0 = alpha * system.t_q + beta * system.t_e[:,:,0] + gamma * system.t_l[:,:,0]
        time_0 = np.repeat(time_0, K+1, axis=0)
        # for type 1
        time_1 = alpha * system.t_q + beta * system.t_e[:,:,1] + gamma * system.t_l[:,:,1]
        time_1 = np.repeat(time_1, K+1, axis=0)
        # for type 2
        time_2 = alpha * system.t_q + beta * system.t_e[:,:,2] + gamma * system.t_l[:,:,2]
        time_2 = np.repeat(time_2, K+1, axis=0)

        time = [time_0, time_1, time_2]
        zeta = [np.dot(-u , (psi*(t)).sum(axis=1)) for t in time]
        self.zeta = ...
        return 

    def __repr__(self):
        return f"TravelerGroup(type={self.traveler_type}, n={len(self.travelers)})"
    
# ==============================================================
# Inidividual Traveler (agent-level state and actions)
# ==============================================================
class Traveler(TravelerBase):
    def __init__(self, 
                 group: TravelerGroup,
                 k_init: int, 
                 d,
                 delta_t: int,
                 T: int,
                 K: int,
                 id: int
                 ):
        super().__init__(group.traveler_type)
        self.group = group
        group.register(self)

        # --- Shared (from group) ---
        self.phi = group.phi
        self.t_star = group.t_star
        self.u = [group.u_low, group.u_high]

        # --- Individual state ---
        self.id = id
        self.k = k_init
        self.u_curr = 0
        self.d = d
        self.t = self.t_star
        self.b = 0
        self.zeta = np.zeros((2, K+1))  # per-agent reward storage
        self.time_slots = [i * delta_t for i in range(T)]



        
        # # change the name of the indices
        # u_k_t_b = [(u,k,t,b) for u in range(U) for k in range(K+1) for t in range(T) for b in range(K+1)]
        # u_k = [(u,k) for u in range(U) for k in range(K+1)]
        # t_b = [(t,b) for t in range(T) for b in range(K+1)]

        # # Compute state transition matrix
        # U = len(self.u)
        # kappa = np.random.rand(U*(K+1)*T*(K+1), K+1)
        # kappa = [kappa_row/np.sum(kappa_row) for kappa_row in kappa]
        # self.P = np.tile(kappa, (1,U))
        # for i in range(U):
        #     for j in range(U):
        #         p = phi[i,j]
        #         idx_i = [val[0] == i for val in u_k_t_b]
        #         idx_j = [val[0] == j for val in u_k]
        #         self.P[idx_i][:,idx_j] *= p

        # # Initialize value function, policiy, etc.
        # self.zeta = np.zeros(U*(K+1)*T*(K+1)) # by type ??
        # self.Q = np.zeros(U*(K+1)*T*(K+1))
        # self.V = np.zeros(U*(K+1)) 
        # self.pi = np.random.rand(U*(K+1), T*(K+1))
        # for i in range(len(u_k)):
        #     k = u_k[i][1]
        #     self.pi[i, k+1:] = 0
        # self.pi = [row/np.sum(row) for row in self.pi]

    def action(self):
        """
        TODO: Define a better logic.
        
        """
        self.t = self.t_star  

        if self.u_curr == 1:
            self.b = self.k
        else:
            self.b = 0
        return 

    def paid_karma_bid(self):
        """Deduct the bid from the traveler's karma balance."""
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
    def __init__(self, fast_lane_capacity: int, slow_lane_capacity: int):

        # --- Fixed attributes ---
        self.fast_lane_capacity = fast_lane_capacity
        self.slow_lane_capacity = slow_lane_capacity
        # avoid magic numbers here
        U = 2  # number of urgency levels
        K = 1000  # maximum karma balance
        T = 36  # number of time slots in a day

        # --- Change  ---
        self.b_star = None            # threshold bid for fast lane
        self.slow_lane_queue:int = 0     # current queue length in slow lane
        # dim2 for urgency levels and dim3 for traveler types
        self.psi: np.ndarray = np.zeros((T*(K+1),2)) # probability of entering fast/slow lane
        self.t_q: np.ndarray = np.zeros((T,2))
        self.t_e: np.ndarray = np.zeros((T,2,3))
        self.t_l: np.ndarray = np.zeros((T,2,3))

        # TODO: might be heavy to store but useful 
        self.fast_lane_travelers: list[Traveler] = []
        self.slow_lane_travelers: list[Traveler] = []

    def karma_redistribution(self):
        """
        TODO: 
        - First redistribute uniformly.
        - Later, redistribute based on traveler states.
        """
        total_karma_used = sum(t.b for t in self.fast_lane_travelers)
        all_travelers = self.fast_lane_travelers + self.slow_lane_travelers
        n_travelers = len(all_travelers)
        if n_travelers > 0:
            karma_per_traveler = total_karma_used // n_travelers
            remainder = total_karma_used % n_travelers  # leftover karma
            for i, traveler in enumerate(all_travelers):
                extra = 1 if i < remainder else 0
                traveler.get_new_karma(karma_per_traveler + extra)

        # Reset for next time slot
        self.fast_lane_travelers = []
        self.slow_lane_travelers = []
        return

    def cost_lane(self, group_travelers: list[list[Traveler]]):
        """
        Loop over each departure time group and compute:
        - who enters the fast vs slow lane,
        - how the queue evolves in the slow lane.

        group_travelers: list of lists, 
        each sublist = travelers with the same departure time.
        """
        costs = []
        self.slow_lane_queue = 0  
        self.psi = np.zeros_like(self.psi) 
        for index, travelers in enumerate(group_travelers):
            if len(travelers) == 0:
                self.b_star = 0
                # Update slow-lane queue
                self.update_slow_lane_queue_length(
                    0
                )
            else:
                departure_time = travelers[0].t
                self.compute_b_star(travelers)


                for traveler in travelers:
                    if traveler.b > self.b_star:
                        self.fast_lane_travelers.append(traveler)
                    else:
                        self.slow_lane_travelers.append(traveler)
                n_slow_lane = len(self.slow_lane_travelers)

                # Update slow-lane queue
                self.update_slow_lane_queue_length(
                    n_slow_lane
                )

                # TODO: compute and record travel costs
                # For now: store b_star and queue info
                costs.append({
                    "time": departure_time,
                    "b_star": self.b_star,
                    "queue_length": self.slow_lane_queue,
                    "n_fast_lane": len(self.fast_lane_travelers),
                    "n_slow_lane": n_slow_lane,
                })
            
            K = 1000  # avoid magic numbers here
            self.psi[index*(K+1)+self.b_star : (index+1)*(K+1),1] = 1 # independant from urgency
            self.t_q[index,0] = self.slow_lane_queue/self.slow_lane_capacity # time spent in the queue
            # type dependent issues
            for traveler in travelers:
                self.t_e[index,:, traveler.traveler_type] = max(0, traveler.t_star - traveler.t - self.t_q[index,:])
                self.t_l[index,:, traveler.traveler_type] = max(0, traveler.t + self.t_q[index,:] - traveler.t_star)

        # Who bid for the fast lane -> determine fast/slow lane assignment
        # Then determine how many people enter the slow lane
        # Update the queue length
        # Repeat these steps for every departure time
        return costs
    
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

        # Basic deterministic queue update
        self.slow_lane_queue = max(0, self.slow_lane_queue - (capacity - inflow))
        return 

    # based on t, b and same state, compute the cost 