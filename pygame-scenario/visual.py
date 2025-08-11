import torch
import numpy as np
import pygame
import math
import time  

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

CAR_LENGTH = 50 
CAR_WIDTH = 30
max_speed = 5
acceleration = 0.1
friction = 0.95
max_steering_angle = 45
wheelbase = 40

SPOT_WIDTH = 80
SPOT_HEIGHT = 80
BORDER_THICKNESS = 5

original_surface = pygame.Surface((CAR_LENGTH, CAR_WIDTH), pygame.SRCALPHA)
original_surface.fill(BLUE)

class ParkingEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.screen = None
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Parking Game")

        spot_x = (SCREEN_WIDTH - SPOT_WIDTH) // 2
        spot_y = BORDER_THICKNESS
        self.parking_spot = pygame.Rect(spot_x, spot_y, SPOT_WIDTH, SPOT_HEIGHT)
        
        self.borders = [
            pygame.Rect(spot_x, spot_y - BORDER_THICKNESS, SPOT_WIDTH, BORDER_THICKNESS),  # Top
            pygame.Rect(spot_x - BORDER_THICKNESS, spot_y, BORDER_THICKNESS, SPOT_HEIGHT),  # Left
            pygame.Rect(spot_x + SPOT_WIDTH, spot_y, BORDER_THICKNESS, SPOT_HEIGHT)       # Right
        ]

        border_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for b in self.borders:
            pygame.draw.rect(border_surface, RED, b)
        self.border_mask = pygame.mask.from_surface(border_surface)

        self.spot_mask = pygame.Mask((SPOT_WIDTH, SPOT_HEIGHT))
        self.spot_mask.fill()

        self.screen_mask = pygame.Mask((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen_mask.fill()

        self.spot_center_x = spot_x + SPOT_WIDTH / 2
        self.spot_center_y = spot_y + SPOT_HEIGHT / 2

        self.max_steps = 3600

        self.max_dist = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)

        self.reset()

    def reset(self):
        self.car_x = SCREEN_WIDTH // 2
        self.car_y = SCREEN_HEIGHT // 2
        self.car_angle = 0  
        self.car_speed = 0.0
        self.steering_angle = 0.0
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        state = [
            self.car_x / SCREEN_WIDTH,
            self.car_y / SCREEN_HEIGHT,
            math.sin(math.radians(self.car_angle)),
            math.cos(math.radians(self.car_angle)),
            self.car_speed / max_speed,
            self.steering_angle / max_steering_angle
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        action_map = [
            (0, 0), (0, -1), (0, 1),
            (-1, 0), (-1, -1), (-1, 1),
            (1, 0), (1, -1), (1, 1)
        ]
        accel_dir, steer_dir = action_map[action]

        up = accel_dir == 1
        down = accel_dir == -1
        left = steer_dir == -1
        right = steer_dir == 1

        if up:
            self.car_speed += acceleration
        if down:
            self.car_speed -= acceleration
        self.car_speed = min(max_speed, max(-max_speed / 2, self.car_speed))
        if not up and not down:
            self.car_speed *= friction

        target_steering = 0
        if left:
            target_steering = max_steering_angle
        if right:
            target_steering = -max_steering_angle
        self.steering_angle += (target_steering - self.steering_angle) * 0.2

        old_angle = self.car_angle

        if abs(self.car_speed) > 0.01:
            turn_rate = (self.car_speed / wheelbase) * math.tan(math.radians(self.steering_angle))
            self.car_angle += math.degrees(turn_rate)

        rotated_surface = pygame.transform.rotate(original_surface, self.car_angle)
        car_rect = rotated_surface.get_rect(center=(self.car_x, self.car_y))
        car_mask = pygame.mask.from_surface(rotated_surface)

        collided = False
        overlap_screen = self.screen_mask.overlap_mask(car_mask, (int(car_rect.left), int(car_rect.top)))
        if overlap_screen.count() != car_mask.count():
            collided = True
        if self.border_mask.overlap(car_mask, (int(car_rect.left), int(car_rect.top))):
            collided = True

        if collided:
            self.car_angle = old_angle
            rotated_surface = pygame.transform.rotate(original_surface, self.car_angle)
            car_rect = rotated_surface.get_rect(center=(self.car_x, self.car_y))
            car_mask = pygame.mask.from_surface(rotated_surface)

        old_x, old_y = self.car_x, self.car_y
        dx = self.car_speed * math.cos(math.radians(self.car_angle))
        dy = -self.car_speed * math.sin(math.radians(self.car_angle))
        self.car_x += dx
        self.car_y += dy
        car_rect = rotated_surface.get_rect(center=(self.car_x, self.car_y))

        collided = False
        overlap_screen = self.screen_mask.overlap_mask(car_mask, (int(car_rect.left), int(car_rect.top)))
        if overlap_screen.count() != car_mask.count():
            collided = True
        if self.border_mask.overlap(car_mask, (int(car_rect.left), int(car_rect.top))):
            collided = True

        reward = -1.0

        if collided:
            self.car_x = old_x
            self.car_y = old_y
            self.car_speed *= -0.5
            reward -= 10.0  

        rel_x = int(car_rect.left - self.parking_spot.left)
        rel_y = int(car_rect.top - self.parking_spot.top)
        overlap_spot = self.spot_mask.overlap_mask(car_mask, (rel_x, rel_y))
        parked = overlap_spot.count() == car_mask.count()

        done = False
        if parked:
            reward += 1000.0  # Success reward
            done = True

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            if not parked:
                reward -= 100.0   

        dist = math.sqrt((self.car_x - self.spot_center_x)**2 + (self.car_y - self.spot_center_y)**2)
        reward -= dist / self.max_dist

        angle_diff = abs((self.car_angle % 360) - 90) / 180.0
        reward -= angle_diff

        if self.render_mode:
            self.render(rotated_surface, car_rect)

        return self.get_state(), reward, done, {}

    def render(self, rotated_surface, car_rect):
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, GREEN, self.parking_spot)
        for border in self.borders:
            pygame.draw.rect(self.screen, RED, border)
        self.screen.blit(rotated_surface, car_rect)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()

class DQN(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.out = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dim = 6
action_dim = 9
q_net = DQN(state_dim, action_dim).to(device)
q_net.load_state_dict(torch.load("parking_dqn_model.pth"))
q_net.eval()

env = ParkingEnv(render_mode=True)

state = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    with torch.no_grad():
        q_values = q_net(torch.tensor(state, device=device).unsqueeze(0))
        action = q_values.max(1)[1].item() 

    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward
    steps += 1

    time.sleep(1 / 60) 

print(f"Visualization complete! Total Reward: {total_reward}, Steps: {steps}")
env.close()