import pygame
import random
import sys
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Parking Game")

# Colors
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Car properties
CAR_LENGTH = 50  # forward direction
CAR_WIDTH = 30
car_x = SCREEN_WIDTH // 2
car_y = SCREEN_HEIGHT // 2
car_angle = 90  # degrees, 90 is up
car_speed = 0.0
max_speed = 5
acceleration = 0.1
friction = 0.95
steer_speed = 3

# Parking spot properties
SPOT_WIDTH = 60
SPOT_HEIGHT = 40
BORDER_THICKNESS = 5

# Font for messages
font = pygame.font.Font(None, 36)

# Function to generate random parking spot and borders
def generate_parking_spot():
    spot_x = random.randint(BORDER_THICKNESS, SCREEN_WIDTH - SPOT_WIDTH - BORDER_THICKNESS)
    spot_y = random.randint(BORDER_THICKNESS, SCREEN_HEIGHT - SPOT_HEIGHT - BORDER_THICKNESS)
    spot = pygame.Rect(spot_x, spot_y, SPOT_WIDTH, SPOT_HEIGHT)
    
    center_x = spot_x + SPOT_WIDTH // 2
    if center_x < SCREEN_WIDTH / 2:
        # Left half, open on right, borders on top, bottom, left
        borders = [
            pygame.Rect(spot_x, spot_y - BORDER_THICKNESS, SPOT_WIDTH, BORDER_THICKNESS),  # Top
            pygame.Rect(spot_x, spot_y + SPOT_HEIGHT, SPOT_WIDTH, BORDER_THICKNESS),      # Bottom
            pygame.Rect(spot_x - BORDER_THICKNESS, spot_y, BORDER_THICKNESS, SPOT_HEIGHT)  # Left
        ]
    else:
        # Right half, open on left, borders on top, bottom, right
        borders = [
            pygame.Rect(spot_x, spot_y - BORDER_THICKNESS, SPOT_WIDTH, BORDER_THICKNESS),  # Top
            pygame.Rect(spot_x, spot_y + SPOT_HEIGHT, SPOT_WIDTH, BORDER_THICKNESS),      # Bottom
            pygame.Rect(spot_x + SPOT_WIDTH, spot_y, BORDER_THICKNESS, SPOT_HEIGHT)       # Right
        ]
    return borders, spot

# Create original car surface
original_surface = pygame.Surface((CAR_LENGTH, CAR_WIDTH), pygame.SRCALPHA)
original_surface.fill(BLUE)

# Main game loop
def main():
    global car_x, car_y, car_speed, car_angle
    clock = pygame.time.Clock()
    running = True
    game_over = False
    win = False

    # Generate initial parking spot and borders
    borders, parking_spot = generate_parking_spot()

    # Start timer
    start_time = pygame.time.get_ticks()
    time_limit = 10000  # 5 seconds in milliseconds

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            keys = pygame.key.get_pressed()

            old_x, old_y = car_x, car_y

            # Acceleration and friction
            if keys[pygame.K_UP]:
                car_speed += acceleration
            if keys[pygame.K_DOWN]:
                car_speed -= acceleration
            car_speed = min(max_speed, max(-max_speed / 2, car_speed))
            if not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
                car_speed *= friction

            # Steering
            if keys[pygame.K_LEFT]:
                car_angle += steer_speed  # left turn (ccw)
            if keys[pygame.K_RIGHT]:
                car_angle -= steer_speed  # right turn (cw)

            # Calculate movement
            dx = car_speed * math.cos(math.radians(car_angle))
            dy = -car_speed * math.sin(math.radians(car_angle))  # negative for y up

            car_x += dx
            car_y += dy

            # Rotate car surface
            rotated_surface = pygame.transform.rotate(original_surface, car_angle)
            car_rect = rotated_surface.get_rect(center=(car_x, car_y))

            # Check collisions with borders
            collided = False
            for border in borders:
                if car_rect.colliderect(border):
                    collided = True
                    break

            # Check screen bounds
            if car_rect.left < 0 or car_rect.right > SCREEN_WIDTH or car_rect.top < 0 or car_rect.bottom > SCREEN_HEIGHT:
                collided = True

            if collided:
                car_x = old_x
                car_y = old_y
                car_speed *= -0.5  # Bounce back a bit

            # Draw parking spot
            pygame.draw.rect(screen, GREEN, parking_spot)

            # Draw borders
            for border in borders:
                pygame.draw.rect(screen, RED, border)

            # Draw car
            screen.blit(rotated_surface, car_rect)

            # Check for success: car entirely within parking spot
            if parking_spot.contains(car_rect):
                game_over = True
                win = True

            # Check timer
            elapsed_time = pygame.time.get_ticks() - start_time
            remaining_time = max(0, (time_limit - elapsed_time) // 1000)
            if elapsed_time > time_limit:
                game_over = True
                win = False

            # Display remaining time
            time_text = font.render(f"Time: {remaining_time}s", True, WHITE)
            screen.blit(time_text, (10, 10))

        else:
            # Game over message
            if win:
                message = font.render("You parked successfully!", True, GREEN)
            else:
                message = font.render("Time's up! You lost.", True, RED)
            screen.blit(message, (SCREEN_WIDTH // 2 - message.get_width() // 2, SCREEN_HEIGHT // 2 - message.get_height() // 2))

            # Restart prompt
            restart_text = font.render("Press R to restart or Q to quit", True, WHITE)
            screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 50))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                # Reset game
                car_x = SCREEN_WIDTH // 2
                car_y = SCREEN_HEIGHT // 2
                car_speed = 0
                car_angle = 90
                borders, parking_spot = generate_parking_spot()
                start_time = pygame.time.get_ticks()
                game_over = False
                win = False
            elif keys[pygame.K_q]:
                running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()