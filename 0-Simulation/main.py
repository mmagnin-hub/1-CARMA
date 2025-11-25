from itertools import groupby
from operator import attrgetter
from entities import TravelerGroup, Traveler, System

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

    # --- Initialize traveler groups ---
    groups = [
        TravelerGroup(p["type"], p["phi"], p["t_star"], p["u_low"], p["u_high"], K, T)
        for p in types
    ]

    # --- Initialize travelers ---
    travelers = []
    id_track = 0
    for group, params in zip(groups, types):
        for i in range(params["count"]):
            travelers.append(
                Traveler(group, k_init, d=None, delta_t=delta_t, T=T, K=K, id=id_track + i)
            )
        id_track += params["count"]

    # --- Initialize system ---
    system = System(fast_lane_capacity=10, slow_lane_capacity=50, K=K, T=T, travelers=travelers) # s = fast lane + slow lane
    # capacity is based on the time slot duration 

    # ==============================================================
    # Simulation loop (3 objects)
    # groups : list[TravelerGroup], 
    # travelers : list[Traveler],
    # system : System
    # ==============================================================
    n_day = 50
    for day in range(n_day):
        # --- Step 1: Each traveler acts and bids ---
        for traveler in travelers:
            traveler.action() 

        # --- Step 2: System reaction ---
        system.simulate_lane_queue() 

        # -- Step 3: Travelers pay their karma bids ---
        for traveler in travelers:
            traveler.paid_karma_bid() 
        
        # -- Step 4: System redistributes karma ---
        system.karma_redistribution() 

        # --- Step 5: Update traveler urgency and policies---
        for traveler in travelers:
            traveler.update_urgency()
        for group in groups:
            group.update_policy()  


        

if __name__ == "__main__":
    main()
