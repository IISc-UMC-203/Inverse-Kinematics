import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Import 3D plotting toolkit
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Tuple, List
import numpy as np

try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
except ImportError:
    print("FrEIA library not found.")
    print("Please install it: pip install FrEIA")
    exit()

# --- Helper Functions for 3D Transformations ---

def translation_matrix(dx, dy, dz, device='cpu'):
    """Creates a 4x4 translation matrix."""
    T = torch.eye(4, device=device).float()
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = dz
    return T

def rotation_matrix_x(angle, device='cpu'):
    """Creates a 4x4 rotation matrix around the X-axis."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.eye(4, device=device).float()
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R

def rotation_matrix_y(angle, device='cpu'):
    """Creates a 4x4 rotation matrix around the Y-axis."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.eye(4, device=device).float()
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R

def rotation_matrix_z(angle, device='cpu'):
    """Creates a 4x4 rotation matrix around the Z-axis."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.eye(4, device=device).float()
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R

# --- RobotArm3d Class ---

class RobotArm3d():
    """
    3D Robot Arm with a defined kinematic structure (e.g., P-R-R-R).
    Includes INN for inverse kinematics.
    Uses homogeneous transformation matrices for forward kinematics.
    """
    def __init__(self, kinematic_structure: list, sigmas: list, viz_dir: str = "visualizations_3d", data_dir: str = "data_3d"):
        """
        Args:
            kinematic_structure (list): A list of tuples, where each tuple defines a joint and the subsequent link.
                                         Format: ('Type', 'Axis', LinkLength)
                                         Type: 'P' (Prismatic), 'R' (Revolute)
                                         Axis: 'X', 'Y', 'Z' (axis of actuation)
                                         LinkLength: Length of the link *following* this joint, measured along the standard axis (usually X) of the new frame.
                                         Example: [('P', 'Z', 0.0), ('R', 'Z', L1), ('R', 'Y', L2), ('R', 'Y', L3)]
            sigmas (list): Standard deviations for sampling each joint variable. Length must match kinematic_structure.
            viz_dir (str): Directory for saving visualizations.
            data_dir (str): Directory for saving data (like the INN model).
        """
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print(f"Using device: {self.device}")

        if len(kinematic_structure) != len(sigmas):
            raise ValueError("Length of kinematic_structure must match length of sigmas.")

        self.kinematic_structure = kinematic_structure
        self.sigmas = torch.tensor(sigmas, device=self.device).float()
        self.num_joints = len(self.kinematic_structure)

        self.pos_dim = 3  # (x, y, z)
        self.latent_dim = self.num_joints - self.pos_dim

        if self.latent_dim < 0:
             raise ValueError(f"Number of joints ({self.num_joints}) must be >= position dimensions ({self.pos_dim})")

        # Extract link lengths for convenience (used in visualization range)
        self.link_lengths = torch.tensor([s[2] for s in kinematic_structure], device=self.device).float()

        # --- Visualization & Output ---
        # Estimate reach based on prismatic range and link lengths
        max_prismatic_reach = 3 * self.sigmas[0].item() if kinematic_structure[0][0] == 'P' else 0 # Rough estimate
        total_reach = self.link_lengths.sum().cpu().item() + abs(max_prismatic_reach)
        plot_margin = 1.2
        self.range_lim = (-total_reach * plot_margin, total_reach * plot_margin)
        self.viz_dir = viz_dir
        self.out_dir = data_dir
        self.inn_model_file = os.path.join(self.out_dir, f"robot_arm_3d_inn_{self.num_joints}j.pt")

        if not os.path.isdir(self.out_dir): os.makedirs(self.out_dir, exist_ok=True)
        if not os.path.isdir(self.viz_dir): os.makedirs(self.viz_dir, exist_ok=True)

        # --- INN Model ---
        self.inn = self._build_inn().to(self.device)
        print(f"INN initialized for {self.num_joints} joints ({self.pos_dim} pos + {self.latent_dim} latent) with {sum(p.numel() for p in self.inn.parameters())} parameters.")
        self._load_inn_model()

    def sample_priors(self, batch_size: int = 1) -> torch.Tensor:
        """Normal distributed values of the joint parameters"""
        # Ensure compatibility with torch.randn_like if needed, or use torch.randn directly
        return torch.randn(batch_size, self.num_joints, device=self.device) * self.sigmas

    def _get_joint_transform(self, joint_idx: int, theta_i: torch.Tensor) -> torch.Tensor:
        """
        Calculates the 4x4 homogeneous transformation matrix for a single joint
        based on its type and axis, given its variable theta_i.
        Handles batching (theta_i has shape [batch_size]).
        Returns tensor of shape [batch_size, 4, 4].
        """
        joint_type, joint_axis, _ = self.kinematic_structure[joint_idx]
        batch_size = theta_i.shape[0]
        transforms = torch.eye(4, device=self.device).float().unsqueeze(0).repeat(batch_size, 1, 1)

        if joint_type == 'P': # Prismatic
            if joint_axis == 'X': transforms[:, 0, 3] = theta_i
            elif joint_axis == 'Y': transforms[:, 1, 3] = theta_i
            elif joint_axis == 'Z': transforms[:, 2, 3] = theta_i
            else: raise ValueError(f"Invalid prismatic axis: {joint_axis}")
        elif joint_type == 'R': # Revolute
            cos_t = torch.cos(theta_i)
            sin_t = torch.sin(theta_i)
            if joint_axis == 'X':
                transforms[:, 1, 1] = cos_t; transforms[:, 1, 2] = -sin_t
                transforms[:, 2, 1] = sin_t; transforms[:, 2, 2] = cos_t
            elif joint_axis == 'Y':
                transforms[:, 0, 0] = cos_t; transforms[:, 0, 2] = sin_t
                transforms[:, 2, 0] = -sin_t; transforms[:, 2, 2] = cos_t
            elif joint_axis == 'Z':
                transforms[:, 0, 0] = cos_t; transforms[:, 0, 1] = -sin_t
                transforms[:, 1, 0] = sin_t; transforms[:, 1, 1] = cos_t
            else: raise ValueError(f"Invalid revolute axis: {joint_axis}")
        else:
            raise ValueError(f"Invalid joint type: {joint_type}")

        return transforms

    def _get_link_transform(self, joint_idx: int) -> torch.Tensor:
         """ Creates the 4x4 homogeneous transformation for the link *following* joint_idx.
             Assumes link length is along the X-axis of the joint's frame.
             Returns tensor of shape [4, 4] (batch-independent).
         """
         _, _, link_length = self.kinematic_structure[joint_idx]
         if link_length == 0.0:
             return torch.eye(4, device=self.device).float()
         else:
             # Assumes link is along X axis of the *current* frame
             return translation_matrix(link_length, 0, 0, device=self.device)


    def forward_kinematics_batch(self, thetas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate all joint positions and the final end-effector position for a batch of configurations.
        Args:
            thetas (torch.Tensor): Batch of joint configurations, shape (batch_size, num_joints).
        Returns:
            all_joint_positions (torch.Tensor): Positions of all joints (including base and end-effector),
                                                shape (batch_size, num_joints + 1, 3).
            end_effector_pos (torch.Tensor): Final end-effector positions, shape (batch_size, 3).
        """
        thetas = thetas.to(self.device).float()
        batch_size = thetas.shape[0]

        # Store T0_i for each joint i (transform from base to joint i's frame)
        # Position of joint i is the origin of frame i expressed in base frame
        all_joint_positions = torch.zeros(batch_size, self.num_joints + 1, 3, device=self.device) # Base (0,0,0) + num_joints ends

        # T_current is the transformation from the base frame to the current joint's frame
        T_current = torch.eye(4, device=self.device).float().unsqueeze(0).repeat(batch_size, 1, 1) # Shape (batch_size, 4, 4)

        for i in range(self.num_joints):
            # 1. Apply transformation for joint i actuation
            T_joint = self._get_joint_transform(i, thetas[:, i]) # Shape (batch_size, 4, 4)
            T_current = torch.bmm(T_current, T_joint) # T_0_i_actuation = T_0_(i-1)_link @ T_(i-1)_i_actuation

            # 2. Apply transformation for link i (connects joint i to joint i+1)
            T_link = self._get_link_transform(i) # Shape (4, 4) - batch independent
            T_current = torch.bmm(T_current, T_link.unsqueeze(0).repeat(batch_size, 1, 1)) # T_0_i_link = T_0_i_actuation @ T_i_act_i_link

            # Store position of the end of link i (which is the origin of frame i+1)
            all_joint_positions[:, i+1, :] = T_current[:, :3, 3] # Extract translation part

        end_effector_pos = all_joint_positions[:, -1, :] # Position of the last joint frame's origin
        return all_joint_positions, end_effector_pos

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """ Forward kinematics: returns only the end-effector position for the batch. """
        _, end_effector_pos = self.forward_kinematics_batch(thetas)
        return end_effector_pos

    # --- INN Methods (largely unchanged logic, only dimensions updated) ---

    def _build_inn(self):
        """Builds the INN model using FrEIA."""
        def subnet_fc(dims_in, dims_out):
            # Consider increasing complexity slightly for 3D if needed
            return nn.Sequential(
                nn.Linear(dims_in, 192), nn.LeakyReLU(0.1), # Increased nodes
                nn.Linear(192, 192), nn.LeakyReLU(0.1),
                nn.Linear(192, dims_out)
            )

        nodes = [Ff.InputNode(self.num_joints, name='input')]
        n_coupling_blocks = 8 # Increased blocks for potentially harder problem
        for k in range(n_coupling_blocks):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name=f'perm_{k}'))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {'subnet_constructor': subnet_fc, 'clamp': 1.9}, name=f'glow_{k}'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        return Ff.GraphINN(nodes)

    def _save_inn_model(self):
        print(f"Saving 3D INN model to {self.inn_model_file}")
        torch.save(self.inn.state_dict(), self.inn_model_file)

    def _load_inn_model(self):
        if os.path.exists(self.inn_model_file):
            print(f"Loading 3D INN model from {self.inn_model_file}")
            try:
                 state_dict = torch.load(self.inn_model_file, map_location='cpu')
                 self.inn.load_state_dict(state_dict)
                 self.inn.to(self.device)
                 self.inn.eval()
                 print("3D INN model loaded successfully.")
            except Exception as e:
                 print(f"Error loading 3D INN model: {e}. Please check architecture or retrain.")
                 print("Proceeding without loaded model.")
        else:
            print("No pre-trained 3D INN model found. Please train the model.")

    def train_inn(self, n_epochs=100, batch_size=512, lr=1e-4, n_samples_train=500000, n_samples_val=50000, weight_fk=10.0, weight_latent=1.0, weight_rev=5.0):
        """Trains the 3D INN model."""
        print("Starting 3D INN training...")
        # (Training loop logic is identical to 2D version, just uses the 3D forward function)
        self.inn.train()
        optimizer = optim.Adam(self.inn.parameters(), lr=lr)
        # Note: verbose parameter is deprecated, but functionally okay for now.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)

        mse_loss = nn.MSELoss()
        print("Generating validation data...")
        with torch.no_grad():
            thetas_val = self.sample_priors(n_samples_val)
            pos_val = self.forward(thetas_val).detach() # Uses 3D forward

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss_f = 0.0
            epoch_loss_z = 0.0
            epoch_loss_rev = 0.0
            epoch_loss_total = 0.0
            n_batches = max(1, n_samples_train // batch_size)

            self.inn.train()
            for i in range(n_batches):
                optimizer.zero_grad()

                # Forward pass
                with torch.no_grad():
                    thetas_batch = self.sample_priors(batch_size)
                    pos_batch_gt = self.forward(thetas_batch).detach() # Uses 3D forward

                out_inn_tensor, _ = self.inn(thetas_batch)
                pos_pred = out_inn_tensor[:, :self.pos_dim] # First 3 are position
                z_pred = out_inn_tensor[:, self.pos_dim:]

                loss_f = mse_loss(pos_pred, pos_batch_gt)
                loss_z = torch.mean(z_pred**2) / 2.0

                # Reverse Pass
                z_rev_sample = torch.randn(batch_size, self.latent_dim, device=self.device)
                rev_input = torch.cat((pos_batch_gt, z_rev_sample), dim=1)
                thetas_rev_pred_tensor, _ = self.inn(rev_input, rev=True)

                pos_rev_pred = self.forward(thetas_rev_pred_tensor) # Uses 3D forward
                loss_rev = mse_loss(pos_rev_pred, pos_batch_gt)

                total_loss = weight_fk * loss_f + weight_latent * loss_z + weight_rev * loss_rev
                total_loss.backward()
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(self.inn.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_loss_f += loss_f.item()
                epoch_loss_z += loss_z.item()
                epoch_loss_rev += loss_rev.item()
                epoch_loss_total += total_loss.item()

                # (Progress printing remains the same)
                if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
                     print(f" Batch {i+1}/{n_batches} - Loss: {total_loss.item():.4f} (Fk:{loss_f.item():.4f}, Z:{loss_z.item():.4f}, Rev:{loss_rev.item():.4f})", end='\r')

            # --- FIX IS HERE: Calculate averages BEFORE printing ---
            avg_epoch_loss_f = epoch_loss_f / n_batches
            avg_epoch_loss_z = epoch_loss_z / n_batches
            avg_epoch_loss_rev = epoch_loss_rev / n_batches
            avg_epoch_loss = epoch_loss_total / n_batches # Now avg_epoch_loss is defined
            print() # Newline after batch progress


            # Validation
            # (Validation loop logic is identical to 2D version, just uses the 3D forward function)
            self.inn.eval()
            with torch.no_grad():
                val_loss_total = 0.0
                n_val_batches = max(1, n_samples_val // batch_size)
                for j in range(n_val_batches):
                    start_idx = j * batch_size
                    end_idx = min((j + 1) * batch_size, n_samples_val)
                    if start_idx >= end_idx: continue

                    thetas_val_batch = thetas_val[start_idx:end_idx]
                    pos_val_batch_gt = pos_val[start_idx:end_idx]
                    current_batch_size = thetas_val_batch.shape[0]

                    # Forward validation
                    out_val_tensor, _ = self.inn(thetas_val_batch)
                    pos_val_pred = out_val_tensor[:, :self.pos_dim]
                    z_val_pred = out_val_tensor[:, self.pos_dim:]
                    loss_f_val = mse_loss(pos_val_pred, pos_val_batch_gt)
                    loss_z_val = torch.mean(z_val_pred**2) / 2.0

                    # Reverse validation
                    z_val_rev_sample = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                    rev_val_input = torch.cat((pos_val_batch_gt, z_val_rev_sample), dim=1)
                    thetas_val_rev_pred_tensor, _ = self.inn(rev_val_input, rev=True)
                    pos_val_rev_pred = self.forward(thetas_val_rev_pred_tensor) # Uses 3D forward
                    loss_rev_val = mse_loss(pos_val_rev_pred, pos_val_batch_gt)

                    total_val_loss_batch = weight_fk * loss_f_val + weight_latent * loss_z_val + weight_rev * loss_rev_val
                    val_loss_total += total_val_loss_batch.item() * current_batch_size

            avg_val_loss = val_loss_total / n_samples_val
            elapsed_time = time.time() - start_time

            # --- Now the print statement will work ---
            print(f"Epoch {epoch+1}/{n_epochs} - Time: {elapsed_time:.1f}s - Train Loss: {avg_epoch_loss:.4f} (F:{avg_epoch_loss_f:.4f} Z:{avg_epoch_loss_z:.4f} R:{avg_epoch_loss_rev:.4f}) - Val Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model.")
                best_val_loss = avg_val_loss
                self._save_inn_model()
            elif epoch > 25 and optimizer.param_groups[0]['lr'] < 1e-6: # Adjusted early stopping patience
                 print("Learning rate too low or plateaued too long, stopping early.")
                 break

        print("3D Training finished.")
        print("Loading best 3D model found during training.")
        self._load_inn_model()
        self.inn.eval()

    def inverse_inn(self, pos_target: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """ Performs inverse kinematics using the trained 3D INN. """
        self.inn.eval()
        n_targets = pos_target.shape[0]
        if pos_target.shape[1] != 3:
             raise ValueError(f"Target position must have 3 dimensions (x, y, z), got shape {pos_target.shape}")
        pos_target = pos_target.to(self.device).float()

        if self.latent_dim <= 0 and n_samples > 1:
             print("Warning: num_joints == pos_dim (3). Only one latent state (zero vector) is possible.")
             n_samples = 1

        with torch.no_grad():
            if self.latent_dim > 0:
                z_samples = torch.randn(n_targets * n_samples, self.latent_dim, device=self.device)
            else:
                 z_samples = torch.empty(n_targets * n_samples, 0, device=self.device)

            pos_target_rep = pos_target.repeat_interleave(n_samples, dim=0)
            # Ensure dimensions match INN input size
            rev_input = torch.cat((pos_target_rep, z_samples), dim=1)
            if rev_input.shape[1] != self.num_joints:
                 raise RuntimeError(f"Dimension mismatch for reverse INN input. Expected {self.num_joints}, got {rev_input.shape[1]}")

            thetas_pred_tensor, _ = self.inn(rev_input, rev=True)

        return thetas_pred_tensor.float()


    # --- Visualization Methods (Updated for 3D) ---

    def init_plot_3d(self) -> Tuple[plt.Figure, plt.Axes]:
        """Initialize matplotlib 3D figure"""
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(*self.range_lim)
        ax.set_ylim(*self.range_lim)
        ax.set_zlim(*self.range_lim) # Set Z limits
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # Set aspect ratio to be equal. 'box' is often better for 3D.
        ax.set_aspect('equal', adjustable='box')
        # Add origin lines
        ax.plot([0, 0], [0, 0], [self.range_lim[0], self.range_lim[1]], 'k:', alpha=0.3) # Z
        ax.plot([0, 0], [self.range_lim[0], self.range_lim[1]], [0, 0], 'k:', alpha=0.3) # Y
        ax.plot([self.range_lim[0], self.range_lim[1]], [0, 0], [0, 0], 'k:', alpha=0.3) # X

        return fig, ax

    def distance_euclidean(self, pos_target: torch.Tensor, pos_pred: torch.Tensor) -> float:
        """Calculates mean Euclidean distance in 3D."""
        # (Logic remains the same as 2D, works for any dimension)
        if pos_pred is None or pos_pred.shape[0] == 0 or pos_target.shape[0] == 0:
            return float('nan')
        if pos_target.shape[1] != 3 or pos_pred.shape[1] != 3:
            print(f"Warning: Mismatched dimensions in distance calc. Target: {pos_target.shape}, Pred: {pos_pred.shape}")
            return float('nan')

        pos_target = pos_target.to(self.device).float()
        pos_pred = pos_pred.to(self.device).float()

        if pos_target.shape[0] == 1 and pos_pred.shape[0] > 1:
            target_expanded = pos_target.expand_as(pos_pred)
        elif pos_target.shape[0] > 1:
             if pos_target.shape[0] == pos_pred.shape[0]:
                 target_expanded = pos_target
             else:
                 n_samples = pos_pred.shape[0] // pos_target.shape[0]
                 if pos_pred.shape[0] % pos_target.shape[0] != 0:
                      print("Warning: Prediction count not multiple of target count in distance calc.")
                      target_expanded = pos_target.mean(dim=0, keepdim=True).expand_as(pos_pred)
                 else:
                      target_expanded = pos_target.repeat_interleave(n_samples, dim=0)
        else:
            return float('nan')

        distances = torch.sqrt(((pos_pred - target_expanded) ** 2).sum(dim=1))
        return distances.mean().item()


    def viz_inverse_3d(self, pos_target: torch.Tensor, thetas: torch.Tensor, max_solutions_to_plot: int = 50, save: bool = True, show: bool = False, fig_name: str = "fig_inverse_inn_3d", viz_format: tuple = (".png", ".svg")):
        """ Visualize inverse kinematics solutions in 3D (Static Plot). """
        if not isinstance(thetas, torch.Tensor):
            raise TypeError(f"Expected 'thetas' to be a torch.Tensor, but got {type(thetas)}")
        if pos_target.shape[1] != 3:
             raise ValueError("Target position must be 3D")

        fig, ax = self.init_plot_3d()

        n_solutions = thetas.shape[0]
        thetas_to_plot = thetas[:max_solutions_to_plot]
        n_plot = thetas_to_plot.shape[0]

        if n_plot == 0:
            print("Warning: No solutions provided for visualization.")
            all_p_final = torch.empty(0, 3, device=self.device)
        else:
            # Calculate FK for the solutions we intend to plot
            all_joint_positions, all_p_final = self.forward_kinematics_batch(thetas_to_plot)
            all_joint_positions_np = all_joint_positions.detach().cpu().numpy() # (n_plot, n_joints+1, 3)
            p_final_np = all_p_final.detach().cpu().numpy() # (n_plot, 3)

            # Plot Arm Segments
            segment_alpha = max(0.02, 0.7 / np.sqrt(n_plot + 1)) # Adjust alpha based on number
            for i in range(n_plot):
                xs = all_joint_positions_np[i, :, 0]
                ys = all_joint_positions_np[i, :, 1]
                zs = all_joint_positions_np[i, :, 2]
                ax.plot(xs, ys, zs, marker='o', markersize=2, linestyle='-', linewidth=1.5, alpha=segment_alpha, color='gray') # Plot full arm trace

            # Plot End Effector Points (more prominently)
            ee_alpha = max(0.1, 0.8 / np.sqrt(n_plot + 1))
            ax.scatter(p_final_np[:, 0], p_final_np[:, 1], p_final_np[:, 2],
                       c='green', marker='o', s=10, alpha=ee_alpha, label=f'End Effectors ({n_plot}/{n_solutions})', zorder=6)

        # Plot target position(s)
        pos_target_np = pos_target.cpu().numpy()
        ax.scatter(pos_target_np[:, 0], pos_target_np[:, 1], pos_target_np[:, 2],
                   marker='x', color='blue', s=200, alpha=1.0, depthshade=False, # depthshade=False keeps marker color constant
                   label='Target(s)', zorder=10)

        # Calculate overall distance using *all* generated thetas if available, otherwise just plotted ones
        if n_solutions > n_plot:
             with torch.no_grad():
                  all_p_final_full = self.forward(thetas) # Calculate EE for all solutions
                  distance = self.distance_euclidean(pos_target, all_p_final_full)
        else:
             distance = self.distance_euclidean(pos_target, all_p_final)


        # Final Plot Styling
        title = f"Inverse Kinematics (3D INN) - {n_plot} sols plotted / {n_solutions} generated"
        if not np.isnan(distance):
            title += f"\nMean Euclidean Distance (all sols) = {distance:.4f}"
        ax.set_title(title)
        if n_plot > 0: ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1)) # Adjust legend position for 3D


        if save:
            for fmt in viz_format:
                fmt_cleaned = fmt if fmt.startswith('.') else '.' + fmt
                full_path = os.path.join(self.viz_dir, fig_name) + fmt_cleaned
                print(f"Saving 3D visualization to {full_path}")
                try:
                    fig.savefig(full_path, bbox_inches='tight', dpi=300)
                except Exception as e:
                    print(f"Error saving 3D figure to {full_path}: {e}")
        if show:
            plt.show()

        plt.close(fig) # Close the figure


    def animate_inverse_solutions_3d(self, pos_target: torch.Tensor, thetas: torch.Tensor, frame_skip: int = 1, interval: int = 50, save: bool = True, show: bool = False, anim_name: str = "anim_inverse_inn_3d", anim_format: str = ".mp4"):
        """Animates the 3D robot arm configurations found for a target position."""
        if not isinstance(thetas, torch.Tensor):
            raise TypeError(f"Expected 'thetas' to be a torch.Tensor, but got {type(thetas)}")
        if pos_target.shape[1] != 3:
             raise ValueError("Target position must be 3D")

        n_solutions = thetas.shape[0]
        if n_solutions == 0:
            print("No solutions to animate.")
            return

        thetas_to_animate = thetas[::frame_skip]
        num_frames = thetas_to_animate.shape[0]
        print(f"Animating {num_frames} configurations (skipped every {frame_skip})...")

        fig, ax = self.init_plot_3d()

        # Plot target position statically
        pos_target_np = pos_target.cpu().numpy()
        ax.scatter(pos_target_np[:, 0], pos_target_np[:, 1], pos_target_np[:, 2],
                   marker='x', color='blue', s=150, alpha=1.0, depthshade=False, label='Target(s)', zorder=10)
        ax.set_title(f"3D INN Solutions Animation (Frame 0/{num_frames})")

        # Initialize plot elements for the arm (one line for the whole arm, one scatter for EE)
        # Using plot for the arm segments
        line, = ax.plot([], [], [], marker='o', markersize=3, linestyle='-', linewidth=2.0, color='black', zorder=5)
        # Using scatter for the end effector
        ee_point = ax.scatter([], [], [], c='red', s=60, zorder=8, label='End Effector', depthshade=False)
        frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes) # Use text2D for fixed position text

        ax.legend(loc='upper right') # Add legend

        # --- Animation Update Function ---
        def update(frame_idx):
            theta_single = thetas_to_animate[frame_idx].unsqueeze(0) # Need batch dim
            # Use forward_kinematics_batch which returns all joint positions
            joint_positions, _ = self.forward_kinematics_batch(theta_single)
            joint_positions_np = joint_positions.squeeze(0).detach().cpu().numpy() # Remove batch dim, move to numpy

            # Update arm segments line
            xs = joint_positions_np[:, 0]
            ys = joint_positions_np[:, 1]
            zs = joint_positions_np[:, 2]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

            # Update end effector position
            # For scatter plot in 3D, need to update _offsets3d
            ee_coords = joint_positions_np[-1:, :] # Keep as (1, 3) shape
            ee_point._offsets3d = (ee_coords[:, 0], ee_coords[:, 1], ee_coords[:, 2])

            # Update title and frame text
            ax.set_title(f"3D INN Solutions Animation (Frame {frame_idx + 1}/{num_frames})")
            frame_text.set_text(f'Solution: {frame_idx * frame_skip + 1} / {n_solutions}')

            # Return list of artists modified
            return line, ee_point, frame_text

        # --- Create and Save/Show Animation ---
        # blit=True is often problematic with 3D plots, use blit=False
        ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                      interval=interval, blit=False, repeat=False)

        if save:
            fmt_cleaned = anim_format if anim_format.startswith('.') else '.' + anim_format
            full_path = os.path.join(self.viz_dir, anim_name) + fmt_cleaned
            print(f"Saving 3D animation to {full_path}...")
            try:
                writer = None
                if fmt_cleaned == '.mp4':
                    writer = animation.FFMpegWriter(fps=1000/interval, bitrate=1800) # Adjust bitrate if needed
                elif fmt_cleaned == '.gif':
                     writer = animation.PillowWriter(fps=1000/interval)
                else:
                    print(f"Warning: Unsupported animation format '{fmt_cleaned}'. Saving may fail.")

                ani.save(full_path, writer=writer, dpi=150)
                print("3D Animation saved successfully.")
            except FileNotFoundError:
                 print("\nERROR: Could not save 3D animation. Ensure FFmpeg (for .mp4) or Pillow (for .gif) is installed and accessible.")
            except Exception as e:
                print(f"Error saving 3D animation to {full_path}: {e}")

        if show:
            plt.show()

        plt.close(fig)


# --- Main Execution Example ---
if __name__ == "__main__":
    TRAIN_MODEL_3D = False   # Set to True to retrain
    ANIMATE_RESULTS_3D = True
    SHOW_STATIC_PLOT_3D = True

    # --- Define 3D Arm Structure ---
    # Example: 4-DOF P-R-R-R Arm (Prismatic Z, Rotate Z, Rotate Y, Rotate Y)
    L1, L2, L3 = 0.8, 0.7, 0.5
    kinematic_structure_3d = [
        ('P', 'Z', 0.0), # Prismatic joint along Z, no link length after it (joint 2 starts here)
        ('R', 'Z', L1),  # Revolute joint around Z, followed by link L1 along new X
        ('R', 'Y', L2),  # Revolute joint around Y, followed by link L2 along new X
        ('R', 'Y', L3)   # Revolute joint around Y, followed by link L3 along new X (End Effector)
    ]
    # Sigmas: std dev for prismatic displacement and angles (radians)
    sigmas_3d = [0.3, 0.8, 1.0, 1.0] # Example sigmas

    # --- Target ---
    target_pos_3d = torch.tensor([[0.5, 0.6, 0.7]]) # Example 3D target

    # --- Parameters ---
    num_inverse_samples_3d = 1000
    animation_frame_skip_3d = 5
    animation_interval_3d = 60 # ms

    # --- Setup ---
    print("\n--- Setting up 3D Robot Arm ---")
    arm_3d = RobotArm3d(kinematic_structure=kinematic_structure_3d, sigmas=sigmas_3d)

    # --- Training ---
    if TRAIN_MODEL_3D or not os.path.exists(arm_3d.inn_model_file):
        print("\n--- Starting 3D INN Training ---")
        arm_3d.train_inn(
            n_epochs=50,         # Might need more epochs for 3D
            batch_size=16384*4,    # Adjust based on GPU memory
            lr=2e-4,             # Potentially slightly lower LR
            n_samples_train=2000000, # More samples might help
            n_samples_val=100000,
            weight_fk=100.0,      # Higher weight on FK reconstruction
            weight_latent=0.8,
            weight_rev=150.0       # Higher weight on reverse mapping
        )
        print("--- 3D Training Complete ---")
    else:
        print("--- Skipping 3D Training (model file exists) ---")
        arm_3d._load_inn_model() # Ensure model is loaded

    # --- Inverse Kinematics ---
    print(f"\n--- Generating {num_inverse_samples_3d} inverse solutions for 3D target: {target_pos_3d.numpy()} ---")
    start_time = time.time()

    # Basic check if model seems ready
    try: next(arm_3d.inn.parameters())
    except StopIteration:
        print("ERROR: 3D INN model appears empty. Cannot perform inverse kinematics.")
        exit()

    thetas_generated_inn_3d = arm_3d.inverse_inn(target_pos_3d, n_samples=num_inverse_samples_3d)
    time_taken = time.time() - start_time
    print(f"Time to generate 3D solutions: {time_taken:.3f} seconds")

    if not isinstance(thetas_generated_inn_3d, torch.Tensor) or thetas_generated_inn_3d.numel() == 0:
        print(f"Error: 3D inverse_inn did not return a valid tensor. Type: {type(thetas_generated_inn_3d)}")
        exit()
    else:
        print(f"Generated {thetas_generated_inn_3d.shape[0]} configurations.")

    # --- Visualization (Static & Animated) ---
    if thetas_generated_inn_3d.shape[0] > 0:
        print("\n--- Processing 3D Results ---")

        # --- Create Filenames ---
        target_name_3d = f"x{target_pos_3d[0,0].item():.1f}_y{target_pos_3d[0,1].item():.1f}_z{target_pos_3d[0,2].item():.1f}"
        target_name_3d = target_name_3d.replace('.', 'p').replace('-', 'm')
        base_filename_3d = f"{arm_3d.num_joints}j_{num_inverse_samples_3d}s_{target_name_3d}"

        # --- Static Visualization ---
        if SHOW_STATIC_PLOT_3D:
            print("Generating static 3D visualization...")
            fig_name_static_3d = f"static_inverse_inn_{base_filename_3d}"
            arm_3d.viz_inverse_3d(
                target_pos_3d,
                thetas_generated_inn_3d,
                max_solutions_to_plot=100, # Limit static plot complexity
                fig_name=fig_name_static_3d,
                show=True,
                save=True
            )

        # --- Animation ---
        if ANIMATE_RESULTS_3D:
            print("Generating 3D animation...")
            anim_name_3d = f"anim_inverse_inn_{base_filename_3d}"
            arm_3d.animate_inverse_solutions_3d(
                target_pos_3d,
                thetas_generated_inn_3d,
                frame_skip=animation_frame_skip_3d,
                interval=animation_interval_3d,
                save=True,
                show=False,
                anim_name=anim_name_3d,
                anim_format=".mp4"
            )

    else:
        print("\n--- Skipping 3D Visualization (No solutions generated) ---")

    print("\n--- 3D Script Finished ---")