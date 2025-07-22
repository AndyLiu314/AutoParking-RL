from parallel_env import ParkingEnv

env = ParkingEnv(render_mode="human")
obs = env.reset()
env.render()

#keep running until the user closes the window
input("Press enter to continue...")

env.close()
