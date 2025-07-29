import gymnasium as gym
import numpy as np
import pygame
from parallel_env import ParkingEnv

class ManualParkingControl:
    def __init__(self):
        """Initialize manual control for parking environment"""
        self.env = ParkingEnv(render_mode="human")
        
        # Configure the environment for manual control
        self.env.configure({
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "speed_range": [-10, 10],  # Allow reverse speeds
                "steering_range": [-np.pi/4, np.pi/4],
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 80,
            "screen_width": 1080,
            "screen_height": 720,
            "centering_position": [0.5, 0.5],
            "scaling": 10,
            "controlled_vehicles": 1,
            "vehicles_count": 3,
            "add_walls": True,
        })
        
        # Print action space info
        print(f"🎯 Action space: {self.env.action_space}")
        print(f"🎯 Action space shape: {self.env.action_space.shape}")
        print(f"🎯 Action space low: {self.env.action_space.low}")
        print(f"🎯 Action space high: {self.env.action_space.high}")
        
        self.obs = None
        self.info = None
        self.running = True
        
        # Control variables
        self.steering = 0.0
        self.acceleration = 0.0
        self.steering_speed = 0.1
        self.acceleration_speed = 0.1  # Reduced from 0.3 for slower movement
        
        print("🎮 Manual Parking Control")
        print("Controls:")
        print("  S - Forward")
        print("  W - Backward")
        print("  Q/E - Alternative Forward/Backward")
        print("  D/A - Steer Left/Right")
        print("  X - Brake (stop)")
        print("  SPACE - Reset")
        print("  ESC - Quit")
        print("  R - Random spawn")
        print("  G - Show goal position")
    
    def handle_events(self):
        """Handle pygame events for manual control"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            if event.type == pygame.KEYDOWN:
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
                    self.acceleration = 0.0  # Brake - stop immediately
                    print("🛑 Brake pressed - stopping!")
        
        # Continuous key handling
        keys = pygame.key.get_pressed()
        
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
        
        # Acceleration control (W/S keys)
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
        print("🔄 Resetting environment...")
        self.obs, self.info = self.env.reset()
        self.steering = 0.0
        self.acceleration = 0.0
    
    def random_spawn(self):
        """Spawn at a random position"""
        print("🎲 Random spawn...")
        # The environment will handle random spawning
        self.reset_environment()
    
    def show_goal_info(self):
        """Show goal position information"""
        if self.obs is not None and isinstance(self.obs, dict):
            if 'desired_goal' in self.obs:
                goal = self.obs['desired_goal']
                print(f"🎯 Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
                print(f"🎯 Goal orientation: cos={goal[4]:.2f}, sin={goal[5]:.2f}")
    
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
                print("🏁 Episode finished!")
                print(f"💰 Final Reward: {self.last_reward:.3f}")
                if self.info.get('is_success', False):
                    print("🎉 SUCCESS! Car parked successfully!")
                self.reset_environment()
            
            # Render
            self.env.render()
        
        self.env.close()
        print("👋 Manual control ended")

def main():
    """Main function to run manual parking control"""
    try:
        controller = ManualParkingControl()
        controller.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the parallel_env module is available")

if __name__ == "__main__":
    main()