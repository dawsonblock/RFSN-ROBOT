"""
Pick and Throw Demo
===================
Robot picks up the cube and throws it!

Run: python pick_and_throw.py
"""

import mujoco
import numpy as np
import time
import glfw

MODEL_PATH = "panda_table_cube.xml"

# PD control gains
KP = np.array([400.0, 400.0, 400.0, 400.0, 200.0, 150.0, 80.0])
KD = np.array([80.0, 80.0, 80.0, 80.0, 40.0, 30.0, 16.0])

# Motion sequence waypoints
WAYPOINTS = [
    # (name, joint_angles, gripper_state, hold_frames, description)
    ("Home", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]), 0.0, 500, "Starting position"),
    ("Above Cube", np.array([0.2, -0.4, 0.0, -1.6, 0.0, 1.2, 0.785]), 0.0, 600, "Moving above cube"),
    ("Approach Cube", np.array([0.2, -0.2, 0.0, -1.2, 0.0, 0.8, 0.785]), 0.0, 400, "Lowering to cube"),
    ("Grasp", np.array([0.2, -0.2, 0.0, -1.2, 0.0, 0.8, 0.785]), 0.08, 1200, "Closing gripper HARD..."),
    ("Lift", np.array([0.2, -0.8, 0.0, -2.2, 0.0, 1.4, 0.785]), 0.08, 800, "Lifting cube high"),
    ("Wind Up", np.array([-1.0, -0.6, -0.3, -1.6, 0.0, 1.0, 0.785]), 0.08, 500, "Winding up..."),
    ("Throw Start", np.array([0.5, 0.0, 0.3, -1.0, 0.5, 1.5, 0.785]), 0.08, 80, "Accelerating..."),
    ("Throw!", np.array([1.4, 0.4, 0.8, -0.5, 1.8, 2.2, 0.785]), 0.0, 50, "THROWING!"),
    ("Follow Through", np.array([1.8, 0.6, 1.2, -0.3, 2.2, 2.6, 0.0]), 0.0, 1000, "Follow through"),
    ("Return Home", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]), 0.0, 1000, "Returning home"),
]

def init_glfw():
    """Initialize GLFW for rendering."""
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    
    window = glfw.create_window(1200, 900, "Pick and Throw Demo", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window

def get_cube_position(data):
    """Get the cube's current position."""
    cube_id = data.body("cube").id
    return data.xpos[cube_id].copy()

def main():
    """Run pick and throw demonstration."""
    print("=" * 70)
    print("PICK AND THROW DEMO")
    print("=" * 70)
    print("\nThe robot will:")
    print("  1. Approach the cube")
    print("  2. Grasp it with the gripper")
    print("  3. Lift it up")
    print("  4. THROW IT!")
    print("\n" + "=" * 70)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Set initial position
    data.qpos[:7] = WAYPOINTS[0][1].copy()
    mujoco.mj_forward(model, data)
    
    initial_cube_pos = get_cube_position(data)
    print(f"\nInitial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
    
    # Initialize rendering
    window = init_glfw()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Camera setup - better angle for viewing throw
    cam = mujoco.MjvCamera()
    cam.azimuth = 135
    cam.elevation = -20
    cam.distance = 3.5
    cam.lookat[:] = [0.0, 0.0, 0.8]
    
    opt = mujoco.MjvOption()
    
    # Motion control
    waypoint_idx = 0
    frames_at_waypoint = 0
    frame_count = 0
    
    current_target = WAYPOINTS[0][1].copy()
    current_gripper = WAYPOINTS[0][2]
    
    print(f"\nStarting: {WAYPOINTS[0][0]} - {WAYPOINTS[0][4]}")
    
    max_cube_height = initial_cube_pos[2]
    throw_detected = False
    
    while not glfw.window_should_close(window) and waypoint_idx < len(WAYPOINTS):
        sim_start = time.time()
        
        # Get current waypoint
        name, target_q, gripper_state, hold_frames, description = WAYPOINTS[waypoint_idx]
        
        # Update target
        current_target = target_q.copy()
        current_gripper = gripper_state
        
        # Check if it's time to move to next waypoint
        if frames_at_waypoint >= hold_frames:
            waypoint_idx += 1
            frames_at_waypoint = 0
            if waypoint_idx < len(WAYPOINTS):
                next_name, _, _, _, next_desc = WAYPOINTS[waypoint_idx]
                print(f"\n>> {next_name}: {next_desc}")
                
                # Special handling for throw
                if "Throw" in next_name:
                    print(f"   🚀 {next_name.upper()} 🚀")
        
        frames_at_waypoint += 1
        
        # Apply PD control to arm
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        error = current_target - q
        tau = KP * error - KD * dq
        
        # During throw, apply MAXIMUM force for fast motion
        if waypoint_idx >= 6 and waypoint_idx <= 8:  # Throw sequence
            tau *= 3.0  # Triple the control effort for explosive throw
        
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        
        # Gripper position control - move fingers to target positions
        # When closed (gripper_state = 0.08), fingers should be at their limits
        # Left finger: moves from 0 to -0.04 (closing)
        # Right finger: moves from 0 to +0.04 (closing)
        target_left = -current_gripper  # Negative for left
        target_right = current_gripper   # Positive for right
        
        # Get current gripper positions
        left_q = data.qpos[7]   # Left finger joint
        right_q = data.qpos[8]  # Right finger joint
        left_dq = data.qvel[7]
        right_dq = data.qvel[8]
        
        # PD control for gripper
        gripper_kp = 1000.0
        gripper_kd = 50.0
        
        left_tau = gripper_kp * (target_left - left_q) - gripper_kd * left_dq
        right_tau = gripper_kp * (target_right - right_q) - gripper_kd * right_dq
        
        data.ctrl[7] = np.clip(left_tau, -100, 100)
        data.ctrl[8] = np.clip(right_tau, -100, 100)
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Track cube
        cube_pos = get_cube_position(data)
        if cube_pos[2] > max_cube_height:
            max_cube_height = cube_pos[2]
        
        # Render
        if frame_count % 2 == 0:
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
        
        glfw.poll_events()
        frame_count += 1
        
        # Maintain real-time
        time_until_next = model.opt.timestep - (time.time() - sim_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
    
    # Final statistics
    final_cube_pos = get_cube_position(data)
    distance_traveled = np.linalg.norm(final_cube_pos[:2] - initial_cube_pos[:2])
    
    print("\n" + "=" * 70)
    print("THROW COMPLETE!")
    print("=" * 70)
    print(f"Initial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
    print(f"Final cube position:   [{final_cube_pos[0]:.3f}, {final_cube_pos[1]:.3f}, {final_cube_pos[2]:.3f}]")
    print(f"Max height reached:    {max_cube_height:.3f} m")
    print(f"Horizontal distance:   {distance_traveled:.3f} m")
    print("=" * 70)
    
    # Keep window open to see result
    print("\nWindow will stay open. Close to exit.")
    while not glfw.window_should_close(window):
        glfw.poll_events()
        time.sleep(0.1)
    
    glfw.terminate()

if __name__ == "__main__":
    main()
