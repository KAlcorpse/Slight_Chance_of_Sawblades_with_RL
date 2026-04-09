import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure
import os

# --- Constants from our Game ---
WIDTH, HEIGHT = 400, 600

class SawbladeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human', None], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super(SawbladeEnv, self).__init__()
        self.render_mode = render_mode
        
        # ACTION SPACE: 6 possible combinations
        # 0: None, 1: Left, 2: Right, 3: Jump, 4: Left+Jump, 5: Right+Jump
        self.action_space = spaces.Discrete(6)
        
        # OBSERVATION SPACE: 19 continuous values (Normalized between -1 and 1 ideally)
        # Player (5): x, y, vel_x, vel_y, jumps_left
        # Closest Coin (2): dx, dy
        # 3 Closest Blades (12): dx, dy, vel_x, vel_y (x3)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(19,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset Player
        self.p_x = WIDTH // 2
        self.p_y = HEIGHT - 40
        self.p_vel_x = 0.0
        self.p_vel_y = 0.0
        self.p_jumps = 2
        
        # Reset Game State
        self.score = 0
        self.time_left = 60.0
        self.speed_mult = 1.0
        self.steps = 0
        
        # Reset Entities
        self.coins = [{'x': random.randint(50, 750), 'y': random.randint(350, 550)}]
        self.blades = [self._spawn_blade()]
        
        return self._get_obs(), {}

    def _spawn_blade(self):
        speed = random.uniform(3, 6)
        angle = random.uniform(math.pi/4, 3*math.pi/4)
        return {
            'x': random.randint(20, 780), 'y': -50,
            'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed
        }

    def _get_obs(self):
        # Normalize player data
        obs = [
            self.p_x / WIDTH, self.p_y / HEIGHT, 
            self.p_vel_x / 10.0, self.p_vel_y / 10.0, 
            self.p_jumps / 2.0
        ]
        
        # Closest coin (relative distance)
        cx, cy = self.coins[0]['x'], self.coins[0]['y']
        obs.extend([(cx - self.p_x) / WIDTH, (cy - self.p_y) / HEIGHT])
        
        # Sort blades by distance
        self.blades.sort(key=lambda b: (b['x']-self.p_x)**2 + (b['y']-self.p_y)**2)
        
        # Get up to 3 closest blades
        for i in range(3):
            if i < len(self.blades):
                b = self.blades[i]
                obs.extend([
                    (b['x'] - self.p_x) / WIDTH, (b['y'] - self.p_y) / HEIGHT,
                    b['vx'] / 10.0, b['vy'] / 10.0
                ])
            else:
                # Pad with zeros if less than 3 blades exist
                obs.extend([0.0, 0.0, 0.0, 0.0])
                
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0 # Survival bonus per frame
        terminated = False
        
        # --- 1. Apply Actions ---
        self.p_vel_x = 0
        if action in [1, 4]: self.p_vel_x = -6 # Left
        if action in [2, 5]: self.p_vel_x = 6  # Right
        
        if action in [3, 4, 5]: # Jump
            if self.p_jumps > 0:
                self.p_vel_y = -10
                self.p_jumps -= 1

        # --- 2. Update Physics ---
        self.p_vel_y += 0.5 # Gravity
        self.p_x = max(0, min(self.p_x + self.p_vel_x, WIDTH - 40))
        self.p_y += self.p_vel_y
        
        if self.p_y >= HEIGHT - 40: # Floor
            self.p_y = HEIGHT - 40
            self.p_vel_y = 0
            self.p_jumps = 2

        self.speed_mult += 0.0001
        self.time_left -= 1/60.0
        if self.time_left <= 0:
            terminated = True
            reward -= 150 # penalty for running out of time

        # --- 3. Coin Collision (AABB approximation) ---
        for c in self.coins[:]:
            if abs(self.p_x + 20 - c['x']) < 35 and abs(self.p_y + 20 - c['y']) < 35:
                self.coins.remove(c)
                self.coins.append({'x': random.randint(50, 750), 'y': random.randint(350, 550)})
                self.time_left += 2.0
                reward += 40.0 # Big reward for getting time!

        # --- 4. Blade Spawning & Physics ---
        if self.steps % 240 == 0: # Spawn blade every 4 seconds roughly
            self.blades.append(self._spawn_blade())

        for b in self.blades[:]:
            b['x'] += b['vx'] * self.speed_mult
            b['y'] += b['vy'] * self.speed_mult
            
            if b['x'] <= 20 and b['vx'] < 0: b['vx'] *= -1
            if b['x'] >= WIDTH-20 and b['vx'] > 0: b['vx'] *= -1
            if b['y'] <= 20 and b['vy'] < 0: b['vy'] *= -1
            if b['y'] >= HEIGHT-20 and b['vy'] > 0: b['vy'] *= -1

            # Collision Check
            dist_sq = (self.p_x + 20 - b['x'])**2 + (self.p_y + 20 - b['y'])**2
            if dist_sq < (20 + 20)**2: # Player radius 20, Blade radius 20
                if self.p_vel_y > 0 and (self.p_y + 40) < b['y'] + 16: # Stomp
                    self.blades.remove(b)
                    reward += 70.0 # HUGE reward for killing a blade
                    self.p_vel_y = -8
                    self.p_jumps = 1
                else:
                    terminated = True
                    reward -= 200.0 # Massive penalty for dying

        # Anti-Camping Penalty
        if self.p_x < 50 or self.p_x > WIDTH - 90:
            reward -= 0.05 # Small penalty for hugging walls

        # Calculate distance to coin
        dist_to_coin = math.sqrt((self.p_x - self.coins[0]['x'])**2 + (self.p_y - self.coins[0]['y'])**2)

        # Small reward for being close to the coin (normalized)
        # This creates a "gravity" effect toward the target
        reward += (1.0 - (dist_to_coin / WIDTH)) * 0.01

        return self._get_obs(), reward, terminated, False, {}

# --- Training Execution ---
if __name__ == "__main__":
    # 1. SETUP: Directories and Environments
    log_dir = "./sawblade_logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = SawbladeEnv() 
    env = Monitor(env) 

    # 2. CALLBACKS: Logging and Early Stopping
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=39, min_evals=5, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=10000, callback_after_eval=stop_train_callback, best_model_save_path=log_dir, verbose=1)

    # 3. MODEL INITIALIZATION (The "Warm Start" Logic)
    existing_model_path = "./sawblade_logs/best_model.zip"

    if os.path.exists(existing_model_path):
        print(f"--- Loading existing brain from {existing_model_path} ---")
        # Load the saved data into a new PPO object
        model = PPO.load(existing_model_path, env=env, tensorboard_log=log_dir)
    else:
        print("--- No previous data found. Creating a fresh brain. ---")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.05, # Higher entropy helps the coward explore
            tensorboard_log=log_dir 
        )

    # 4. EXECUTION: Start training
    print("Training in progress...")
    model.learn(total_timesteps=500000, callback=eval_callback)
    model.save("sawblade_ppo_model_final")
    print("Training Complete!")