import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
 


def plot_policy_convergence(error_vec, n_day, n_groups):
    # Keep only the rows actually used
    error_used = error_vec[n_day:, :]  

    # Reverse vertically so the first iteration is at the left
    error_rev = error_used[::-1, :]

    plt.figure(figsize=(10, 5))
    for g in range(n_groups):
        plt.plot(error_rev[:, g], label=f"Group t*={g}")

    plt.title("Policy Convergence Error (Lower is Better)")
    plt.xlabel("Day")
    plt.ylabel("Policy L2 Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_final_policies(groups, n_groups):
    plt.figure(figsize=(15, 10))
    for g in range(n_groups):
        plt.subplot(2, 3, g + 1)
        plt.imshow(groups[g].pi + 1e-12, aspect="auto", norm=LogNorm())
        plt.title(f"Policy for Group {groups[g].traveler_type}")
        plt.xlabel("Action index (t × b)")
        plt.ylabel("State index (u × k)")
        plt.colorbar(label="Policy Probability (log scale)")
    plt.tight_layout()
    plt.show()

# def plot_final_policies_linear(groups, n_groups):
#     plt.figure(figsize=(15, 10))
#     for g in range(n_groups):
#         plt.subplot(2, 3, g + 1)
#         plt.imshow(groups[g].pi, aspect="auto", cmap='inferno') # cmap to enhance visibility
#         plt.title(f"Policy for Group {groups[g].traveler_type}")
#         plt.xlabel("Action index (t × b)")
#         plt.ylabel("State index (u × k)")
#         plt.colorbar(label="Policy Probability")
#     plt.tight_layout()
#     plt.show()

def plot_final_policies_linear(groups, n_groups):
    plt.figure(figsize=(15, 10))
    
    for g in range(n_groups):
        plt.subplot(2, 3, g + 1)
        
        pi = groups[g].pi
        
        plt.imshow(
            pi,
            aspect="auto",
            cmap="viridis",        # brighter colormap
            vmin=0,                # ensures full brightness range
            vmax=np.max(pi)        # avoids washed-out scaling
        )
        
        plt.title(f"Policy for Group {groups[g].traveler_type}")
        plt.xlabel("Action index (t × b)")
        plt.ylabel("State index (u × k)")
        plt.colorbar(label="Policy Probability")
    
    plt.tight_layout()
    plt.show()


def plot_specific_state_policy(groups, n_groups, K, specific_u, specific_k):
    state_index = specific_u * (K + 1) + specific_k

    gap = 2  # visual gap between blocks

    plt.figure(figsize=(10, 5))

    for g in range(n_groups):
        plt.subplot(1, 2, g + 1)

        prob_compressed = []
        action_pos = []
        action_labels = []

        non_zero_idx = np.where(groups[g].pi[state_index, :] > 0)[0]

        current_x = 0
        prev_idx = None

        for idx in non_zero_idx:
            if prev_idx is not None and idx != prev_idx + 1:
                current_x += gap  # compress zero region

            prob_compressed.append(groups[g].pi[state_index, idx])
            action_pos.append(current_x)
            action_labels.append(idx)   # ORIGINAL action index

            current_x += 1
            prev_idx = idx

        plt.bar(action_pos, prob_compressed)
        plt.yscale('log')
        plt.ylim(1e-6, 1)

        # restore original meaning of x-axis
        plt.xticks(action_pos, action_labels, rotation=90)

        plt.title(
            f"Policy for Group {groups[g].traveler_type} "
            f"at State (u={specific_u}, k={specific_k})"
        )
        plt.xlabel("Action index (t × b)")
        plt.ylabel("Action Probability (log scale)")

    plt.tight_layout()
    plt.show()

def plot_specific_state_policy_linear(groups, n_groups, K, specific_u, specific_k):
    state_index = specific_u * (K + 1) + specific_k

    gap = 2  # visual gap between blocks

    plt.figure(figsize=(10, 5))

    for g in range(n_groups):
        plt.subplot(1, 2, g + 1)

        prob_compressed = []
        action_pos = []
        action_labels = []

        non_zero_idx = np.where(groups[g].pi[state_index, :] > 0)[0]

        current_x = 0
        prev_idx = None

        for idx in non_zero_idx:
            if prev_idx is not None and idx != prev_idx + 1:
                current_x += gap  # compress zero region

            prob_compressed.append(groups[g].pi[state_index, idx])
            action_pos.append(current_x)
            action_labels.append(idx)   # ORIGINAL action index

            current_x += 1
            prev_idx = idx

        plt.bar(action_pos, prob_compressed)

        # fixed linear probability scale for all groups
        plt.ylim(0, 1)

        # restore original meaning of x-axis
        plt.xticks(action_pos, action_labels, rotation=90)

        plt.title(
            f"Policy for Group {groups[g].traveler_type} "
            f"at State (u={specific_u}, k={specific_k})"
        )
        plt.xlabel("Action index (t × b)")
        plt.ylabel("Action Probability")

    plt.tight_layout()
    plt.show()
