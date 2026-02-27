import torch
import torch.nn as nn
import math
import time

class MicroBEMNA_V2(nn.Module):
    def __init__(self, grid_size=(20, 20, 20)):
        super().__init__()
        self.grid_size = grid_size
        self.num_points = grid_size[0] * grid_size[1] * grid_size[2]
        
        self.D = nn.Parameter(torch.ones(self.num_points, dtype=torch.float32), requires_grad=False)
        self.gamma = 0.05   # Decay rate
        self.beta = 0.5     # Voltage pull
        self.max_steps = 5000

    def forward(self, start_coords, end_coords, temperature=1.0):
        start_idx = self._coords_to_idx(start_coords)
        end_idx = self._coords_to_idx(end_coords)
        
        # Pass temperature to the Flash Phase
        path, flux = self.flash_probe(start_coords, end_coords, temperature)
        
        if path[-1] == end_idx: 
            self.flow_reinforcement(path, flux)
            
        return path

    def flash_probe(self, start_coords, end_coords, temperature):
        current_coords = start_coords
        path = [self._coords_to_idx(current_coords)]
        
        # Clamp T so we don't divide by zero
        T = max(temperature, 0.05)
        
        for _ in range(self.max_steps):
            if current_coords == end_coords:
                break
                
            neighbors = self._get_valid_neighbors(current_coords)
            probabilities = []
            neighbor_indices = []
            
            for nx, ny, nz in neighbors:
                n_idx = self._coords_to_idx((nx, ny, nz))
                neighbor_indices.append(n_idx)
                
                conductance = self.D[n_idx].item()
                dist = math.sqrt((nx - end_coords[0])**2 + (ny - end_coords[1])**2 + (nz - end_coords[2])**2)
                voltage_pull = math.exp(-self.beta * dist)
                
                probabilities.append(conductance * voltage_pull)
            
            prob_tensor = torch.tensor(probabilities, dtype=torch.float32)
            if prob_tensor.sum() == 0:
                break
                
            # Apply Entropic Temperature logic
            prob_tensor = torch.pow(prob_tensor, 1.0 / T)
            
            # Fallback if the math breaks from massive exponents
            if torch.isinf(prob_tensor).any() or torch.isnan(prob_tensor).any():
                prob_tensor = torch.ones_like(prob_tensor) 
                
            prob_tensor = prob_tensor / prob_tensor.sum()
            
            chosen_index = torch.multinomial(prob_tensor, 1).item()
            next_coords = neighbors[chosen_index]
            path.append(neighbor_indices[chosen_index])
            current_coords = next_coords
            
        flux = 500.0 / len(path) if current_coords == end_coords else 0.0
        return path, flux

    def flow_reinforcement(self, path, flux):
        self.D.data.sub_(self.gamma * self.D.data)
        self.D.data.clamp_(min=0.1)
        path_tensor = torch.tensor(path, dtype=torch.long)
        self.D.data[path_tensor] += flux

    def _get_valid_neighbors(self, coords):
        x, y, z = coords
        moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        valid = []
        for dx, dy, dz in moves:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < self.grid_size[0] and 
                0 <= ny < self.grid_size[1] and 
                0 <= nz < self.grid_size[2]):
                valid.append((nx, ny, nz))
        return valid

    def _coords_to_idx(self, coords):
        x, y, z = coords
        return x * (self.grid_size[1] * self.grid_size[2]) + y * self.grid_size[2] + z

# --- EXECUTION LOOP ---
def run_esf_v2_test():
    GRID_SIZE = (20, 20, 20)
    EPOCHS = 50
    model = MicroBEMNA_V2(grid_size=GRID_SIZE)

    start_point = (2, 2, 2)
    end_point = (18, 18, 18)

    print("--- RUNNING BEMNA V2: ENTROPIC CRYSTALLIZATION ---")
    
    # T starts at 1.0 (high entropy/exploration) and decays to 0.1 (low entropy/crystallization)
    initial_temp = 1.0
    final_temp = 0.1
    
    for epoch in range(1, EPOCHS + 1):
        # Linear temperature decay
        current_temp = initial_temp - ((initial_temp - final_temp) * (epoch / EPOCHS))
        
        path = model(start_point, end_point, temperature=current_temp)
        steps = len(path)
        
        status = "GROUNDED" if path[-1] == model._coords_to_idx(end_point) else "FAILED"
        
        if epoch <= 10 or epoch % 10 == 0:
            print(f"[Epoch {epoch:02d}] Temp: {current_temp:.2f} | Status: {status} | Steps: {steps}")

if __name__ == "__main__":
    run_esf_v2_test()
