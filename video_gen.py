import pygame
import random
import sys
import math
import numpy as np
import os  # <-- Added for directory management
import cv2 # <-- Added for video generation
from stable_baselines3 import PPO

# --- Constants ---
WIDTH, HEIGHT = 400, 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)       
BLUE = (50, 150, 255)     
GOLD = (255, 215, 0)      
GRAY = (40, 40, 40)       

# --- Helper to translate Game State for the AI ---
def get_ai_observation(p_x, p_y, p_vel_x, p_vel_y, p_jumps, coins, blades):
    """Exactly matches the _get_obs() function from training"""
    obs = [
        p_x / WIDTH, p_y / HEIGHT, 
        p_vel_x / 10.0, p_vel_y / 10.0, 
        p_jumps / 2.0
    ]
    
    # Closest coin
    if coins:
        cx, cy = coins[0].x, coins[0].y
        obs.extend([(cx - p_x) / WIDTH, (cy - p_y) / HEIGHT])
    else:
        obs.extend([0.0, 0.0])
        
    # Sort blades by distance
    sorted_blades = sorted(blades, key=lambda b: (b.x - p_x)**2 + (b.y - p_y)**2)
    
    # Get up to 3 closest blades
    for i in range(3):
        if i < len(sorted_blades):
            b = sorted_blades[i]
            obs.extend([
                (b.x - p_x) / WIDTH, (b.y - p_y) / HEIGHT,
                b.base_vel_x / 10.0, b.base_vel_y / 10.0
            ])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])
            
    return np.array(obs, dtype=np.float32)

# --- Classes ---
class Player:
    def __init__(self):
        self.width, self.height = 40, 40
        self.x = WIDTH // 2
        self.y = HEIGHT - self.height
        self.vel_x, self.vel_y = 0.0, 0.0
        self.speed = 6
        self.gravity = 0.5
        self.jump_power = -10
        self.jumps_left = 2
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def apply_ai_action(self, action):
        """Translates the AI's integer (0-5) into movement"""
        self.vel_x = 0
        if action in [1, 4]: self.vel_x = -self.speed
        if action in [2, 5]: self.vel_x = self.speed
        
        if action in [3, 4, 5]: # Jump
            if self.jumps_left > 0:
                self.vel_y = self.jump_power
                self.jumps_left -= 1

    def update(self):
        self.vel_y += self.gravity
        self.x += self.vel_x
        self.y += self.vel_y

        if self.y >= HEIGHT - self.height:
            self.y = HEIGHT - self.height
            self.vel_y = 0
            self.jumps_left = 2 

        self.x = max(0, min(self.x, WIDTH - self.width))
        self.rect.topleft = (self.x, self.y)

    def draw(self, surface):
        pygame.draw.rect(surface, BLUE, self.rect)

class Sawblade:
    def __init__(self):
        self.radius = 20
        self.x = random.randint(self.radius, WIDTH - self.radius)
        self.y = -50 
        speed = random.uniform(3, 6)
        angle = random.uniform(math.pi/4, 3*math.pi/4) 
        self.base_vel_x = math.cos(angle) * speed
        self.base_vel_y = math.sin(angle) * speed

    def update(self, speed_multiplier):
        self.x += self.base_vel_x * speed_multiplier
        self.y += self.base_vel_y * speed_multiplier

        if self.x - self.radius <= 0 or self.x + self.radius >= WIDTH:
            self.base_vel_x *= -1
            self.x = max(self.radius, min(self.x, WIDTH - self.radius))
            
        if self.y - self.radius <= 0 or self.y + self.radius >= HEIGHT:
            self.base_vel_y *= -1
            self.y = max(self.radius, min(self.y, HEIGHT - self.radius))

    def draw(self, surface):
        pygame.draw.circle(surface, RED, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), self.radius // 2)

class Coin:
    def __init__(self):
        self.radius = 15
        self.respawn()

    def respawn(self):
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(HEIGHT - 210, HEIGHT - 50) 
        self.rect = pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius*2, self.radius*2)

    def draw(self, surface):
        pygame.draw.circle(surface, GOLD, (int(self.x), int(self.y)), self.radius)

def main():
    # --- 1. LOAD THE AI MODEL ---
    print("Loading AI Brain...")
    try:
        model = PPO.load("./sawblade_logs/best_model")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Playing: A Slight Chance of RL")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 35)

    # --- VIDEO RECORDER SETUP ---
    output_dir = "rollout"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "ai_playing.mp4")
    
    # mp4v codec is universally compatible with .mp4 extensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))
    print(f"Recording started. Video will be saved to: {video_path}")

    player = Player()
    sawblades = [Sawblade()]
    coins = [Coin()]

    score = 0
    time_left = 60.0
    speed_multiplier = 1.0
    blade_spawn_delay, blade_timer = 4.0, 0.0
    coin_spawn_delay, coin_timer = 6.0, 0.0

    running = True
    game_over = False

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_over:
                # Clean up current recording before restarting
                video_writer.release()
                main() 
                return

        if not game_over:
            speed_multiplier += 0.0001
            time_left -= dt

            if time_left <= 0:
                time_left = 0
                game_over = True

            blade_timer += dt
            if blade_timer >= blade_spawn_delay:
                sawblades.append(Sawblade())
                blade_timer = 0.0
                blade_spawn_delay = max(0.6, blade_spawn_delay * 0.95)

            coin_timer += dt
            if coin_timer >= coin_spawn_delay:
                if len(coins) < 4: coins.append(Coin())
                coin_timer = 0.0
                coin_spawn_delay = max(2.5, coin_spawn_delay * 0.98)

            # --- 2. ASK THE AI FOR THE NEXT MOVE ---
            obs = get_ai_observation(
                player.x, player.y, player.vel_x, player.vel_y, player.jumps_left, coins, sawblades
            )
            action, _states = model.predict(obs, deterministic=True)
            player.apply_ai_action(action.item())
            
            # --- 3. RUN PHYSICS ---
            player.update()
            for blade in sawblades:
                blade.update(speed_multiplier)

            for c in coins[:]:
                if player.rect.colliderect(c.rect):
                    score += 1
                    time_left += 2.0  
                    coins.remove(c)
            if len(coins) == 0: coins.append(Coin())

            for blade in sawblades[:]:
                dist_sq = (player.x + 20 - blade.x)**2 + (player.y + 20 - blade.y)**2
                if dist_sq < (20 + blade.radius)**2:
                    if player.vel_y > 0 and (player.y + 40) < blade.y + (blade.radius * 0.8):
                        sawblades.remove(blade)
                        score += 3 
                        player.vel_y = player.jump_power * 0.85 
                        player.jumps_left = 1 
                    else:
                        game_over = True

        # --- 4. RENDER TO SCREEN ---
        screen.fill(GRAY)
        player.draw(screen)
        for c in coins: c.draw(screen)
        for blade in sawblades: blade.draw(screen)

        score_text = font.render(f"Score: {score}", True, WHITE)
        time_text = font.render(f"Time: {max(0, int(time_left))}s", True, WHITE)
        screen.blit(score_text, (20, 20))
        screen.blit(time_text, (WIDTH - 150, 20))

        if game_over:
            game_over_text = font.render("AI DIED - Press R to Restart", True, RED)
            text_rect = game_over_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            screen.blit(game_over_text, text_rect)

        # --- CAPTURE & WRITE FRAME TO VIDEO ---
        # 1. Grab array representation of the current window surface
        frame_surface = pygame.surfarray.array3d(screen)
        # 2. Pygame works in (width, height, channels). Transpose to match OpenCV's (height, width, channels)
        frame_surface = np.transpose(frame_surface, (1, 0, 2))
        # 3. Pygame uses RGB, OpenCV uses BGR. Swap channels.
        frame_bgr = cv2.cvtColor(frame_surface, cv2.COLOR_RGB2BGR)
        # 4. Save frame to file buffer
        video_writer.write(frame_bgr)

        pygame.display.flip()

    # Clean up and save file safely on exit
    video_writer.release()
    print("Video file saved and finalized.")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()