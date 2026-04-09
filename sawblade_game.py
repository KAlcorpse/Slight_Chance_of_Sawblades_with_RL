import pygame
import random
import sys
import math

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

# --- Classes ---
class Player:
    def __init__(self):
        self.width = 40
        self.height = 40
        self.x = WIDTH // 2
        self.y = HEIGHT - self.height
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 6
        self.gravity = 0.5
        self.jump_power = -10
        self.jumps_left = 2
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def move(self, keys):
        self.vel_x = 0
        if keys[pygame.K_LEFT]:
            self.vel_x = -self.speed
        if keys[pygame.K_RIGHT]:
            self.vel_x = self.speed

    def jump(self):
        if self.jumps_left > 0:
            self.vel_y = self.jump_power
            self.jumps_left -= 1

    def update(self):
        self.vel_y += self.gravity
        self.y += self.vel_y
        self.x += self.vel_x

        if self.y >= HEIGHT - self.height:
            self.y = HEIGHT - self.height
            self.vel_y = 0
            self.jumps_left = 2 

        if self.x < 0:
            self.x = 0
        if self.x > WIDTH - self.width:
            self.x = WIDTH - self.width

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
        current_vel_x = self.base_vel_x * speed_multiplier
        current_vel_y = self.base_vel_y * speed_multiplier

        self.x += current_vel_x
        self.y += current_vel_y

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
        self.x = 0
        self.y = 0
        self.rect = pygame.Rect(0,0,0,0)
        self.respawn()

    def respawn(self):
        # CONSTRAINT: X is kept away from walls, Y is locked to the lower half of the screen
        # This ensures the player can always reach it with a standard or double jump.
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(HEIGHT - 210, HEIGHT - 50) 
        self.rect = pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius*2, self.radius*2)

    def draw(self, surface):
        pygame.draw.circle(surface, GOLD, (int(self.x), int(self.y)), self.radius)

def check_collision(player_rect, circle_x, circle_y, radius):
    closest_x = max(player_rect.left, min(circle_x, player_rect.right))
    closest_y = max(player_rect.top, min(circle_y, player_rect.bottom))

    distance_x = circle_x - closest_x
    distance_y = circle_y - closest_y

    return (distance_x ** 2 + distance_y ** 2) < (radius ** 2)

# --- Main Game Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("A Slight Chance of RL - Escalation Edition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)

    player = Player()
    
    # Use lists to manage multiple active entities
    sawblades = [Sawblade()]
    coins = [Coin()]

    score = 0
    time_left = 60.0
    speed_multiplier = 1.0

    # --- Dynamic Spawning Timers ---
    blade_spawn_delay = 4.0 # Starts at 4 seconds
    blade_timer = 0.0

    coin_spawn_delay = 6.0  # Starts at 6 seconds
    coin_timer = 0.0

    running = True
    game_over = False

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and not game_over:
                    player.jump()
                if event.key == pygame.K_r and game_over:
                    main()
                    return

        if not game_over:
            # 1. Update Game Difficulty Variables
            speed_multiplier += 0.0001
            time_left -= dt

            if time_left <= 0:
                time_left = 0
                game_over = True

            # 2. Dynamic Blade Spawner
            blade_timer += dt
            if blade_timer >= blade_spawn_delay:
                sawblades.append(Sawblade())
                blade_timer = 0.0
                # Reduce delay by 5% each spawn, capping at a blisteringly fast 0.6 seconds
                blade_spawn_delay = max(0.6, blade_spawn_delay * 0.95)

            # 3. Dynamic Coin Spawner
            coin_timer += dt
            if coin_timer >= coin_spawn_delay:
                # Cap max coins on screen to avoid infinite time farming
                if len(coins) < 4: 
                    coins.append(Coin())
                coin_timer = 0.0
                # Coins also spawn slightly faster over time to keep up with the chaos
                coin_spawn_delay = max(2.5, coin_spawn_delay * 0.98)

            # 4. Input and Movement
            keys = pygame.key.get_pressed()
            player.move(keys)
            player.update()

            for blade in sawblades:
                blade.update(speed_multiplier)

            # 5. Player vs Coins
            for c in coins[:]:
                if player.rect.colliderect(c.rect):
                    score += 1
                    time_left += 2.0  
                    coins.remove(c)
                    
            # Ensure there is ALWAYS at least one coin on screen
            if len(coins) == 0:
                coins.append(Coin())

            # 6. Player vs Sawblades
            for blade in sawblades[:]:
                if check_collision(player.rect, blade.x, blade.y, blade.radius):
                    if player.vel_y > 0 and player.rect.bottom < blade.y + (blade.radius * 0.8):
                        sawblades.remove(blade)
                        score += 3 
                        player.vel_y = player.jump_power * 0.85 
                        player.jumps_left = 1 
                    else:
                        game_over = True

        # --- Rendering ---
        screen.fill(GRAY)
        player.draw(screen)
        
        for c in coins:
            c.draw(screen)
        for blade in sawblades:
            blade.draw(screen)

        score_text = font.render(f"Score: {score}", True, WHITE)
        time_text = font.render(f"Time: {max(0, int(time_left))}s", True, WHITE)
        
        # Display the current difficulty metrics (Optional, good for debugging)
        # diff_text = font.render(f"Blade Spawn: {blade_spawn_delay:.1f}s", True, (150, 150, 150))
        # screen.blit(diff_text, (20, 60))

        screen.blit(score_text, (20, 20))
        screen.blit(time_text, (WIDTH - 150, 20))

        if game_over:
            game_over_text = font.render("GAME OVER - Press R to Restart", True, RED)
            text_rect = game_over_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            screen.blit(game_over_text, text_rect)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()