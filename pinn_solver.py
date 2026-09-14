import torch
import torch.nn as nn
import numpy as np

# 1. Define a standard Neural Network to approximate the solution u(x)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Simple dense architecture: 1 Input (x) -> Hidden Layers -> 1 Output (u)
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),        # Tanh activation function is standard for PINNs
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Define the Physics Loss function
# Example: Solving the simple differential equation: du/dx = cos(x)
def compute_physics_loss(model, x_domain):
    x_domain.requires_grad_(True)
    u_pred = model(x_domain)
    
    # Calculate the derivative du/dx using PyTorch AutoGrad
    du_dx = torch.autograd.grad(
        u_pred, x_domain, 
        grad_outputs=torch.ones_like(u_pred), 
        create_graph=True
    )
    
    # The actual physics equation target: du/dx - cos(x) = 0
    physics_residual = du_dx - torch.cos(x_domain)
    
    # Return Mean Squared Error of the residual
    return torch.mean(physics_residual ** 2)

# 3. Setup and Training Loop
if __name__ == "__main__":
    print("Initializing Physics-Informed Neural Network Template...")
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Generate random points in our domain x ∈ [0, 2π]
    x_train = torch.tensor(np.linspace(0, 2*np.pi, 100), dtype=torch.float32).view(-1, 1)
    
    # Quick training demo for 50 epochs
    for epoch in range(50):
        optimizer.zero_grad()
        loss = compute_physics_loss(model, x_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Scientific ML Physics Loss: {loss.item():.6f}")
            
    print("Training demo complete! Script running perfectly.")
