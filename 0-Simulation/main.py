from entities import TravelerGroup, Traveler, System
from plots import plot_policy_convergence, plot_final_policies, plot_specific_state_policy, plot_specific_state_policy_linear
import numpy as np
import pickle

def main():
    # -------------------------------------------------------------
    # 1. Define model dimensions and parameters
    # -------------------------------------------------------------
    U = 2                 
    # K = 200
    T = 10            
    delta_t = 1           
    n_travelers = 100
    # KZ: preset initial karma balance
    K = 50
    k_init = 2

    # Group preference = t_star ∈ {0,1,...,T-1}
    n_groups = 1
    t_star = 6
    phi = np.array([[0.7, 0.3],
                    [0.9, 0.1]])

    u_value = np.array([1.0, 6.0]) # from the paper
    delta = 0.9
    eta = 0.1
    alpha = 1 # queueing weight
    beta = 0.35 # early arrival weight
    gamma = 2 # late arrival weight

    u_value_2 = np.array([1.0, 12.0])  
    alpha_2 = 1
    beta_2 = 0.1
    gamma_2 = 5
    # -------------------------------------------------------------
    # 2. Create traveler groups
    # -------------------------------------------------------------
    groups = []
    
    g1 = TravelerGroup(
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
    groups.append(g1)

    # g2 = TravelerGroup(
    #     type_id=1,
    #     phi=phi,
    #     t_star=t_star,
    #     u_value=u_value_2,
    #     K=K,
    #     T=T,
    #     delta=delta,
    #     eta=eta,
    #     alpha=alpha_2,
    #     beta=beta_2,
    #     gamma=gamma_2
    # )
    # groups.append(g2)

    # -------------------------------------------------------------
    # 3. Create travelers and split across groups
    # -------------------------------------------------------------
    travelers = []
    per_group = n_travelers // n_groups

    traveler_id = 0
    for group in groups:
        for _ in range(per_group):
            # k_init = K // n_travelers
            traveler = Traveler(group=group, k_init=k_init, delta_t=delta_t, id=traveler_id)
            travelers.append(traveler)
            traveler_id += 1

    # -------------------------------------------------------------
    # 4. Initialize the System with all travelers
    # -------------------------------------------------------------
    total_capacity = n_travelers // (T-1)
    fast_lane_capacity = total_capacity // 3
    slow_lane_capacity = total_capacity - fast_lane_capacity

    system = System(
        fast_lane_capacity=fast_lane_capacity,
        slow_lane_capacity=slow_lane_capacity,
        K=K,
        T=T,
        travelers=travelers
    )

    # -------------------------------------------------------------
    # 5. Simulation loop
    # -------------------------------------------------------------
    threshold = 1e-3
    n_day = 100

    # For storing old policies: (states × actions × groups)
    pi_old = np.zeros((U*(K+1), T*(K+1), n_groups))
    error_vec = np.zeros((n_day, n_groups))

    converge = False

    while not converge and n_day > 0:
        print("----- Remaining day", n_day, "-----")
        
        for g in groups:
            pi_old[:, :, g.traveler_type] = g.pi.copy()

        converge = True
        n_day -= 1

        # 1. Travelers act
        for tr in travelers:
            tr.store_start_state()
            tr.action()

        # 2. System queues
        system.simulate_lane_queue()

        # 3. Payment
        for tr in travelers:
            tr.paid_karma_bid()

        # 4. Redistribution
        system.karma_redistribution()

        # 5. Update urgency
        for tr in travelers:
            tr.update_urgency()

        # Update each group (independent policies)
        for g in groups:
            g.update_transition_matrix()
            g.update_policy(system)

        # 6. Convergence
        for g in groups:
            err = np.linalg.norm(g.pi - pi_old[:, :, g.traveler_type])
            print(f"Group {g.traveler_type} error:", err)
            error_vec[n_day, g.traveler_type] = err
            if err > threshold:
                converge = False

    # -------------------------------------------------------------
    # 6. Download results
    # -------------------------------------------------------------
    with open("groups.pkl", "wb") as f:
        pickle.dump(groups, f)

    with open("error_vec.pkl", "wb") as f:
        pickle.dump(error_vec, f)

    with open("simulation_params.pkl", "wb") as f:
        pickle.dump((n_day, n_groups, K, n_travelers), f)

    with open("system.pkl", "wb") as f:
        pickle.dump(system, f)


if __name__ == "__main__":
    main()

