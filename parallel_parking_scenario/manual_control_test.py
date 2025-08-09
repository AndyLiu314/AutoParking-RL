import gymnasium as gym
import numpy as np
import pygame
from parallel_env import ParkingEnv

class ManualParkingControl:
    def __init__(self):
        """Initialize manual control for parking environment"""
        # Initialize pygame for keyboard input
        pygame.init()

        self.env = ParkingEnv(render_mode="human")

        # Print action space info
        print(f"Action space: {self.env.action_space}")
        print(f"Action space shape: {self.env.action_space.shape}")
        print(f"Action space low: {self.env.action_space.low}")
        print(f"Action space high: {self.env.action_space.high}")
        
        self.obs = None
        self.info = None
        self.running = True
        
        # Control variables
        self.steering = 0.0
        self.acceleration = 0.0
        self.steering_speed = 0.1
        self.acceleration_speed = 0.01  # Reduced from 0.3 for slower movement
        self.emergency_brake = False # Flag to maintain braking
        
        print("Manual Parking Control")
        print("Controls:")
        print("  S - Forward")
        print("  W - Backward")
        print("  Q/E - Alternative Forward/Backward")
        print("  D/A - Steer Left/Right")

    def handle_events(self):
        """Handle pygame events for manual control"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            if event.type == pygame.KEYDOWN:
                print(f"Key pressed: {event.key}")  # Debug: print any key press
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.reset_environment()
                elif event.key == pygame.K_r:
                    self.random_spawn()
                elif event.key == pygame.K_g:
                    self.show_goal_info()
                elif event.key == pygame.K_x:
                    # Emergency brake: use maximum negative acceleration
                    self.acceleration = -1.0  # Maximum brake
                    self.emergency_brake = True  # Flag to maintain braking
                    print("Emergency brake pressed - maximum deceleration!")

        # Continuous key handling
        keys = pygame.key.get_pressed()
        
        # Emergency brake handling
        if self.emergency_brake:
            # Maintain maximum braking until stopped
            self.acceleration = -1.0
            # Check if car is stopped (you can access vehicle speed from observation)
            if self.obs is not None and isinstance(self.obs, dict) and 'achieved_goal' in self.obs:
                vx, vy = self.obs['achieved_goal'][2], self.obs['achieved_goal'][3]
                speed = (vx**2 + vy**2)**0.5
                if speed < 0.1:  # If speed is very low, stop emergency braking
                    self.emergency_brake = False
                    print("Emergency brake released - car stopped!")

        # Steering control (A/D keys)
        if keys[pygame.K_d]:  # Left
            self.steering = max(self.steering - self.steering_speed, -1.0)
        elif keys[pygame.K_a]:  # Right
            self.steering = min(self.steering + self.steering_speed, 1.0)
        else:
            # Return to center
            if self.steering > 0:
                self.steering = max(self.steering - self.steering_speed * 0.5, 0)
            elif self.steering < 0:
                self.steering = min(self.steering + self.steering_speed * 0.5, 0)
        
        # Acceleration control (W/S keys) - only if not emergency braking
        if not self.emergency_brake:
            if keys[pygame.K_s]:  # Forward
                self.acceleration = min(self.acceleration + self.acceleration_speed, 1.0)
            elif keys[pygame.K_w]:  # Backward
                self.acceleration = max(self.acceleration - self.acceleration_speed, -1.0)
            elif keys[pygame.K_q]:  # Alternative Forward
                self.acceleration = min(self.acceleration + self.acceleration_speed, 1.0)
            elif keys[pygame.K_e]:  # Alternative Backward
                self.acceleration = max(self.acceleration - self.acceleration_speed, -1.0)
            else:
                # Return to neutral
                if self.acceleration > 0:
                    self.acceleration = max(self.acceleration - self.acceleration_speed * 0.5, 0)
                elif self.acceleration < 0:
                    self.acceleration = min(self.acceleration + self.acceleration_speed * 0.5, 0)
        
        return True
    
    def reset_environment(self):
        """Reset the environment"""
        print("Resetting environment...")
        self.obs, self.info = self.env.reset()
        self.steering = 0.0
        self.acceleration = 0.0
        self.emergency_brake = False # Reset emergency brake flag
    
    def random_spawn(self):
        """Spawn at a random position"""
        print("Random spawn...")
        # The environment will handle random spawning
        self.reset_environment()
    
    def show_goal_info(self):
        """Show goal position information"""
        if self.obs is not None and isinstance(self.obs, dict):
            if 'desired_goal' in self.obs:
                goal = self.obs['desired_goal']
                print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
                print(f"Goal orientation: cos={goal[4]:.2f}, sin={goal[5]:.2f}")

    def get_action(self):
        """Get the current action based on manual input"""
        # Convert steering and acceleration to action space
        # Action format: [acceleration, steering]
        # acceleration: positive = forward, negative = backward
        # steering: positive = right, negative = left
        action = np.array([self.acceleration, self.steering])
        
        return action
    
    def display_info(self):
        """Display current reward information"""
        # Only show info at the end of episodes
        pass
    
    def run(self):
        """Main game loop"""
        self.reset_environment()
        
        while self.running:
            # Handle input
            if not self.handle_events():
                break
            
            # Get action from manual input
            action = self.get_action()
            
            # Step the environment
            self.obs, reward, done, truncated, self.info = self.env.step(action)
            
            # Store reward for display
            self.last_reward = reward
            
            # Display info
            self.display_info()
            
            # Check if episode is done
            if done or truncated:
                print("Episode finished!")
                print(f"Final Reward: {self.last_reward:.3f}")
                if self.info.get('is_success', False):
                    print("SUCCESS! Car parked successfully!")
                self.reset_environment()
            
            # Render
            self.env.render()
        
        self.env.close()
        print("Manual control ended")

def main():
    """Main function to run manual parking control"""
    try:
        controller = ManualParkingControl()
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the parallel_env module is available")

if __name__ == "__main__":
    main()