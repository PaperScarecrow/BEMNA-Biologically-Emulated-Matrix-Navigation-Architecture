import torch
import torch.nn as nn
import time

class BEMNA_V7_2_PhaseSpace(nn.Module):
    def __init__(self, grid_size=(20, 20, 20)):
        super().__init__()
        self.grid_size = grid_size
        self.num_points = grid_size[0] * grid_size[1] * grid_size[2]
        
        self.D = nn.Parameter(torch.zeros((1, 6, 6, *grid_size), dtype=torch.float32), requires_grad=False)
        self.gamma = 0.02
        
        # Initialize void
        self.D.data += 0.01 
        for i in range(6):
            self.D.data[0, i, i, :, :, :] += 0.1 

        self.moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    def flash_wave(self, start_coords, end_coords, max_iterations=300, inference=False):
        phi = torch.zeros((1, 6, *self.grid_size), dtype=torch.float32, device=self.D.device)
        sx, sy, sz = start_coords
        ex, ey, ez = end_coords
        
        phi[0, :, sx, sy, sz] = 1.0 
        history = []
        
        for i in range(max_iterations):
            history.append(phi.clone())
            
            # THE FIX: Prevent signal decay during training so it can find the target and build the tube
            active_D = self.D if inference else self.D + 0.95
            phi_out = torch.einsum('boixyz,bixyz->boxyz', active_D, phi)
            
            phi_new = torch.zeros_like(phi)
            phi_new[:, 0, 1:, :, :] = phi_out[:, 0, :-1, :, :]
            phi_new[:, 1, :-1, :, :] = phi_out[:, 1, 1:, :, :]
            phi_new[:, 2, :, 1:, :] = phi_out[:, 2, :, :-1, :]
            phi_new[:, 3, :, :-1, :] = phi_out[:, 3, :, 1:, :]
            phi_new[:, 4, :, :, 1:] = phi_out[:, 4, :, :, :-1]
            phi_new[:, 5, :, :, :-1] = phi_out[:, 5, :, :, 1:]
            
            if inference:
                phi_new[phi_new < 0.1] = 0.0
                
            phi = torch.clamp(phi_new, 0.0, 1.0)
            
            if phi[0, :, ex, ey, ez].sum() > 0.01:
                history.append(phi.clone())
                break
                
        return history

    def return_stroke(self, start_coords, end_coords, history):
        path = [end_coords]
        current = end_coords
        
        for t in range(len(history)-1, -1, -1):
            if current == start_coords:
                break
                
            phi_t = history[t][0, :, current[0], current[1], current[2]]
            best_dir = torch.argmax(phi_t).item()
            voltage = phi_t[best_dir].item()
            
            if voltage == 0: 
                break 
                
            dx, dy, dz = self.moves[best_dir]
            prev_coords = (current[0] - dx, current[1] - dy, current[2] - dz)
            
            path.append(prev_coords)
            current = prev_coords
            
        path.reverse()
        return path

    def forward(self, start_coords, end_coords, inference=False):
        history = self.flash_wave(start_coords, end_coords, inference=inference)
        path = self.return_stroke(start_coords, end_coords, history)
        
        if not inference and len(path) > 1 and path[0] == start_coords and path[-1] == end_coords:
            self.flow_reinforcement(path)
            
        return path

    def flow_reinforcement(self, path):
        self.D.data.sub_(self.gamma * self.D.data)
        self.D.data.clamp_(min=0.01)
        
        flux = 50.0 / len(path)
        
        for i in range(1, len(path)-1):
            prev_c, curr_c, next_c = path[i-1], path[i], path[i+1]
            
            in_vec = (curr_c[0]-prev_c[0], curr_c[1]-prev_c[1], curr_c[2]-prev_c[2])
            in_d = self.moves.index(in_vec)
            
            out_vec = (next_c[0]-curr_c[0], next_c[1]-curr_c[1], next_c[2]-curr_c[2])
            out_d = self.moves.index(out_vec)
            
            self.D.data[0, out_d, in_d, curr_c[0], curr_c[1], curr_c[2]] += flux


def run_true_generation_test():
    vocab = {
        "The": (2, 2, 2), "cat": (5, 12, 18), "A": (2, 18, 2),
        "dog": (5, 8, 18), "sat": (10, 10, 10), "down": (18, 18, 18), "up": (18, 2, 18)
    }
    model = BEMNA_V7_2_PhaseSpace()
    epochs = 40
    seq_A = ["The", "cat", "sat", "down"]
    seq_B = ["A", "dog", "sat", "up"]
    
    print("[*] PHASE 1: REINFORCING LOGIC TUBES...")
    for epoch in range(1, epochs + 1):
        for s in [seq_A, seq_B]:
            for i in range(len(s) - 1):
                model(vocab[s[i]], vocab[s[i+1]], inference=False)

    print("\n[*] PHASE 2: TRUE GENERATIVE ROUTING (Lateral Inhibition)")
    
    def generate_sequence(start_word, steps=200):
        start_coords = vocab[start_word]
        phi = torch.zeros((1, 6, 20, 20, 20), dtype=torch.float32, device=model.D.device)
        
        phi[0, :, start_coords[0], start_coords[1], start_coords[2]] = 1.0 
        
        words_hit = [start_word]
        
        for i in range(steps):
            phi_out = torch.einsum('boixyz,bixyz->boxyz', model.D, phi)
            phi_new = torch.zeros_like(phi)
            
            phi_new[:, 0, 1:, :, :] = phi_out[:, 0, :-1, :, :]
            phi_new[:, 1, :-1, :, :] = phi_out[:, 1, 1:, :, :]
            phi_new[:, 2, :, 1:, :] = phi_out[:, 2, :, :-1, :]
            phi_new[:, 3, :, :-1, :] = phi_out[:, 3, :, 1:, :]
            phi_new[:, 4, :, :, 1:] = phi_out[:, 4, :, :, :-1]
            phi_new[:, 5, :, :, :-1] = phi_out[:, 5, :, :, 1:]
            
            # THE FIX: Action Potential Threshold raised to 0.5. 
            # This makes the void (0.11) lethal, but tubes (>1.0) survive.
            max_vals, max_indices = phi_new.max(dim=1, keepdim=True)
            mask = max_vals > 0.5 
            
            phi = torch.zeros_like(phi_new)
            phi.scatter_(1, max_indices, mask.to(phi.dtype))
            
            for word, coords in vocab.items():
                if word not in words_hit:
                    if phi[0, :, coords[0], coords[1], coords[2]].sum() >= 1.0:
                        words_hit.append(word)
                        
        return words_hit

    import time
    start_time = time.time()
    print(f"[*] Input: 'The' -> Emergent Flow: {' -> '.join(generate_sequence('The'))}")
    print(f"[*] Input: 'A'   -> Emergent Flow: {' -> '.join(generate_sequence('A'))}")
    print(f"[*] Generation Time: {time.time() - start_time:.4f}s")

if __name__ == "__main__":
    run_true_generation_test()
