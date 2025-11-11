from itertools import groupby
from operator import attrgetter
from entities import Traveler, System

def main():
    # ==============================================================
    # Define types of travelers and system
    # ==============================================================
    delta_t = 5  # time slot duration in minutes
    T = 36  # number of time slots in a day from 0 to T-1 => 3 hours
    time_slots = [i*delta_t for i in range(T)]  # time slots in minutes
    k_init = 10 # initial karma for each traveler
    N = 100  # total number of travelers
    K = N * k_init  # total initial karma in the system <=> maximum number of karma a user can have
    
    types = [
        {"type": 0, "phi": [[0.8,0.2],[0.8,0.2]], "t_star": 8, "u_low": 10, "u_high": 20, "count": 50},
        {"type": 1, "phi": [[0.8,0.2],[0.8,0.2]], "t_star": 8, "u_low": 15, "u_high": 30, "count": 30},
        {"type": 2, "phi": [[0.8,0.2],[0.8,0.2]], "t_star": 9, "u_low": 20, "u_high": 40, "count": 20},
    ]

    # Initialize travelers of each type
    travelers = []
    id_track = 0
    for params in types:
        travelers += [
            Traveler(
                u_low=params["u_low"],
                u_high=params["u_high"],
                phi=params["phi"],
                t_star=params["t_star"],
                k_init=k_init,
                d=None,
                delta_t=delta_t,
                T=T,
                K=K,
                id=i,
                traveler_type=params["type"]
            )
            for i in range(id_track, id_track + params["count"])
        ]
        id_track += params["count"]

    # Initialize system
    system = System(fast_lane_capacity=10, slow_lane_capacity=50) # s = fast lane + slow lane
    # !!! capacity is based on the time slot duration !!! ->yes

    # ==============================================================
    # Simulation loop
    # ==============================================================
    n_day = 50
    for n in range(n_day):
        # --- Step 1: Each traveler acts and bids ---
        for traveler in travelers:
            traveler.action() 
        
        # Sort travelers by departure time
        travelers.sort(key=attrgetter("t"))
        group_travelers = [[] for _ in range(len(time_slots))]
        for t_value, group in groupby(travelers, key=attrgetter("t")):
            group_travelers[t_value] = list(group)

        # --- Step 2: System reaction ---
        costs = system.cost_lane(group_travelers)

        for traveler in travelers:
            if traveler.b > system.b_star:  # took the fast lane
                traveler.paid_karma_bid() 
        
        system.karma_redistribution() # travelers get directly new karma here

        # --- Step 3: Update traveler states and policies ---
        traveler.update_policy() # aggregate by types (outside of the loop)
        for traveler in travelers:
            traveler.update_urgency()
        

if __name__ == "__main__":
    main()
