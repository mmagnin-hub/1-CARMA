from entities import TravelerGroup, Traveler, System
import numpy as np
import random

def main():
    # -------------------------------------------------------------
    # 1. Define model dimensions and parameters
    # -------------------------------------------------------------
    U = 3                  # number of urgency levels
    K = 100                # max karma
    T = 6                  # number of departure time slots
    delta_t = 1            # duration of each time slot

    n_types = 1            # number of traveler groups
    n_travelers = 2       # total number of travelers

    # -------------------------------------------------------------
    # 2. Create group-level parameters
    # -------------------------------------------------------------
    phi = np.array([       # urgency transition matrix (U×U)
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])

    u_value = np.array([1.0, 2.0, 3.0])  # utility weights for urgency levels
    t_star = 4                           # preferred time
    delta = 0.9                          # discount factor
    eta = 0.1                            # smoothing parameter for policy
    alpha = beta = gamma = 42            # penalty parameters

    # -------------------------------------------------------------
    # 3. Create TravelerGroup(s)
    # -------------------------------------------------------------
    groups = []

    group = TravelerGroup(
        type_id=0,
        phi=phi,
        t_star=t_star,
        u_value=u_value,
        K=K,
        T=T,
        delta=delta,
        eta=eta,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    groups.append(group)

    # -------------------------------------------------------------
    # 4. Create Travelers and assign them to the group
    # -------------------------------------------------------------
    travelers = []
    for i in range(n_travelers):
        k_init = K // n_travelers # MM : if there is a remainder ? -> ignored for now
        traveler = Traveler(
            group=group,
            k_init=k_init,
            delta_t=delta_t,
            id=i
        )
        travelers.append(traveler)

    # -------------------------------------------------------------
    # 5. Initialize the System
    # -------------------------------------------------------------
    fast_lane_capacity = 5
    slow_lane_capacity = 10

    system = System(
        fast_lane_capacity=fast_lane_capacity,
        slow_lane_capacity=slow_lane_capacity,
        K=K,
        T=T,
        travelers=travelers
    )



    # ==============================================================
    # Simulation loop (3 objects)
    # groups : list[TravelerGroup], 
    # travelers : list[Traveler],
    # system : System
    # ==============================================================
    n_day = 50
    for day in range(n_day):
        print("----- Day", day, "-----")
        # --- Step 1: Each traveler acts and bids ---
        for traveler in travelers:
            traveler.store_start_state() 
            traveler.action() 
            print("Traveler ID:", traveler.id, "Action t:", traveler.t, " b:", traveler.b, " Urgency:", traveler.u_curr, " Karma:", traveler.k_curr)

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
            group.update_transition_matrix()
            group.update_policy(system)  


        

if __name__ == "__main__":
    main()
