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

CAR_LENGTH = 50  # forward direction
CAR_WIDTH = 30
car_x = SCREEN_WIDTH // 2
car_y = SCREEN_HEIGHT // 2
car_angle = 0  # degrees, 0 is right (sideways)
car_speed = 0.0
max_speed = 5
acceleration = 0.1
friction = 0.95
steer_speed = 3
steering_angle = 0.0
max_steering_angle = 45
wheelbase = 40

SPOT_WIDTH = 80
SPOT_HEIGHT = 80
BORDER_THICKNESS = 5

font = pygame.font.Font(None, 36)

def generate_parking_spot():
    spot_x = (SCREEN_WIDTH - SPOT_WIDTH) // 2
    spot_y = BORDER_THICKNESS
    spot = pygame.Rect(spot_x, spot_y, SPOT_WIDTH, SPOT_HEIGHT)
    
    # Open on bottom, borders on top, left, right
    borders = [
        pygame.Rect(spot_x, spot_y - BORDER_THICKNESS, SPOT_WIDTH, BORDER_THICKNESS),  # Top
        pygame.Rect(spot_x - BORDER_THICKNESS, spot_y, BORDER_THICKNESS, SPOT_HEIGHT),  # Left
        pygame.Rect(spot_x + SPOT_WIDTH, spot_y, BORDER_THICKNESS, SPOT_HEIGHT)       # Right
    ]

    # Create border mask
    border_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    for b in borders:
        pygame.draw.rect(border_surface, RED, b)
    border_mask = pygame.mask.from_surface(border_surface)

    # Create spot mask
    spot_mask = pygame.Mask((SPOT_WIDTH, SPOT_HEIGHT))
    spot_mask.fill()

    return borders, spot, border_mask, spot_mask

# Create original car surface
original_surface = pygame.Surface((CAR_LENGTH, CAR_WIDTH), pygame.SRCALPHA)
original_surface.fill(BLUE)

# Main game loop
def main():
    global car_x, car_y, car_speed, car_angle, steering_angle
    clock = pygame.time.Clock()
    running = True
    game_over = False
    win = False

    # Generate initial parking spot and borders
    borders, parking_spot, border_mask, spot_mask = generate_parking_spot()

    # Create screen mask
    screen_mask = pygame.Mask((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen_mask.fill()

    # Start timer
    start_time = pygame.time.get_ticks()
    time_limit = 60000  # 60 seconds in milliseconds

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            keys = pygame.key.get_pressed()

            # Acceleration and friction
            if keys[pygame.K_UP]:
                car_speed += acceleration
            if keys[pygame.K_DOWN]:
                car_speed -= acceleration
            car_speed = min(max_speed, max(-max_speed / 2, car_speed))
            if not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
                car_speed *= friction

            target_steering = 0
            if keys[pygame.K_LEFT]:
                target_steering = max_steering_angle
            if keys[pygame.K_RIGHT]:
                target_steering = -max_steering_angle
            steering_angle += (target_steering - steering_angle) * 0.2

            old_angle = car_angle

            if abs(car_speed) > 0.01:
                turn_rate = (car_speed / wheelbase) * math.tan(math.radians(steering_angle))
                car_angle += math.degrees(turn_rate)

            rotated_surface = pygame.transform.rotate(original_surface, car_angle)
            car_rect = rotated_surface.get_rect(center=(car_x, car_y))
            car_mask = pygame.mask.from_surface(rotated_surface)

            collided = False
            overlap_screen = screen_mask.overlap_mask(car_mask, (car_rect.left, car_rect.top))
            if overlap_screen.count() != car_mask.count():
                collided = True
            if border_mask.overlap(car_mask, (car_rect.left, car_rect.top)):
                collided = True

            if collided:
                car_angle = old_angle
                rotated_surface = pygame.transform.rotate(original_surface, car_angle)
                car_rect = rotated_surface.get_rect(center=(car_x, car_y))
                car_mask = pygame.mask.from_surface(rotated_surface)

            old_x, old_y = car_x, car_y

            dx = car_speed * math.cos(math.radians(car_angle))
            dy = -car_speed * math.sin(math.radians(car_angle))  # negative for y up

            car_x += dx
            car_y += dy

            car_rect = rotated_surface.get_rect(center=(car_x, car_y))

            collided = False
            overlap_screen = screen_mask.overlap_mask(car_mask, (car_rect.left, car_rect.top))
            if overlap_screen.count() != car_mask.count():
                collided = True
            if border_mask.overlap(car_mask, (car_rect.left, car_rect.top)):
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

            # Check for success: car entirely within parking spot using masks
            rel_x = car_rect.left - parking_spot.left
            rel_y = car_rect.top - parking_spot.top
            overlap_spot = spot_mask.overlap_mask(car_mask, (rel_x, rel_y))
            if overlap_spot.count() == car_mask.count():
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
                car_angle = 0
                steering_angle = 0
                borders, parking_spot, border_mask, spot_mask = generate_parking_spot()
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