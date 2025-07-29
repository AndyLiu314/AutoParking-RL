import numpy as np
from parallel_parking_scenario.parallel_env import ParkingEnv

def simple_test():
    """Simple test to see if vehicle can move backward"""
    
    env = ParkingEnv(render_mode="human")
    
    print("🚗 Testing vehicle movement...")
    
    # Reset
    obs, info = env.reset()
    
    # Try forward
    print("🧪 Testing FORWARD movement...")
    for i in range(10):
        obs, reward, done, truncated, info = env.step(np.array([1.0, 0.0]))
        if isinstance(obs, dict) and 'observation' in obs:
            x, y, vx, vy = obs['observation'][0:4]
            speed = np.sqrt(vx**2 + vy**2)
            print(f"   Step {i+1}: Speed = {speed:.2f}, Position = ({x:.2f}, {y:.2f})")
        env.render()
    
    # Try backward
    print("\n🧪 Testing BACKWARD movement...")
    for i in range(10):
        obs, reward, done, truncated, info = env.step(np.array([-1.0, 0.0]))
        if isinstance(obs, dict) and 'observation' in obs:
            x, y, vx, vy = obs['observation'][0:4]
            speed = np.sqrt(vx**2 + vy**2)
            print(f"   Step {i+1}: Speed = {speed:.2f}, Position = ({x:.2f}, {y:.2f})")
        env.render()
    
    env.close()

if __name__ == "__main__":
    simple_test()