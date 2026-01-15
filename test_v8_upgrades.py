"""
V8 Upgrade Tests: Task-Space MPC and Impedance Control
=======================================================
Tests for new v8 features:
- Task-space receding horizon MPC
- Impedance controller for contact-rich manipulation
- Integration with harness and RFSN
"""

import numpy as np
import mujoco as mj


def test_task_space_mpc_module_import():
    """Test that task-space MPC module can be imported."""
    print("\n" + "=" * 70)
    print("TEST: Task-Space MPC Module Import")
    print("=" * 70)
    
    try:
        from rfsn.mpc_task_space import (
            TaskSpaceRecedingHorizonMPC,
            TaskSpaceMPCConfig,
            TaskSpaceMPCResult
        )
        print("✓ Task-space MPC module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import task-space MPC module: {e}")
        return False


def test_task_space_mpc_config():
    """Test task-space MPC configuration."""
    print("\n" + "=" * 70)
    print("TEST: Task-Space MPC Configuration")
    print("=" * 70)
    
    try:
        from rfsn.mpc_task_space import TaskSpaceMPCConfig
        
        # Test default config
        config = TaskSpaceMPCConfig()
        assert config.H_min == 5
        assert config.H_max == 30
        assert config.max_iterations == 100
        assert config.time_budget_ms == 50.0
        assert config.warm_start == True
        
        print("✓ Task-space MPC config created with defaults")
        
        # Test custom config
        config_custom = TaskSpaceMPCConfig(
            H_min=8,
            H_max=25,
            max_iterations=50,
            time_budget_ms=30.0,
            learning_rate=0.03
        )
        assert config_custom.H_min == 8
        assert config_custom.learning_rate == 0.03
        
        print("✓ Task-space MPC config created with custom parameters")
        return True
    except Exception as e:
        print(f"✗ Failed to create task-space MPC config: {e}")
        return False


def test_task_space_mpc_solver():
    """Test task-space MPC solver basic functionality."""
    print("\n" + "=" * 70)
    print("TEST: Task-Space MPC Solver")
    print("=" * 70)
    
    try:
        from rfsn.mpc_task_space import TaskSpaceRecedingHorizonMPC, TaskSpaceMPCConfig
        
        # Load model
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        
        # Create solver
        config = TaskSpaceMPCConfig(
            H_min=5,
            H_max=10,
            max_iterations=20,
            time_budget_ms=100.0
        )
        solver = TaskSpaceRecedingHorizonMPC(model, config)
        print("✓ Task-space MPC solver created")
        
        # Prepare test inputs
        q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        qd = np.zeros(7)
        x_target_pos = np.array([0.4, 0.0, 0.5])
        x_target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        dt = 0.002
        
        decision_params = {
            'horizon_steps': 8,
            'Q_pos_task': np.ones(3) * 100.0,
            'Q_ori_task': np.ones(3) * 10.0,
            'Q_vel_task': np.ones(6) * 5.0,
            'R_diag': 0.01 * np.ones(7),
            'terminal_Q_pos': np.ones(3) * 200.0,
            'terminal_Q_ori': np.ones(3) * 20.0,
            'du_penalty': 0.01
        }
        
        # Solve
        result = solver.solve(q, qd, x_target_pos, x_target_quat, dt, decision_params)
        
        # Check result
        assert result is not None
        assert result.q_ref_next is not None
        assert result.qd_ref_next is not None
        assert result.q_ref_next.shape == (7,)
        assert result.qd_ref_next.shape == (7,)
        assert result.solve_time_ms > 0
        assert result.iters > 0
        
        print(f"✓ Task-space MPC solve completed")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iters}")
        print(f"  Solve time: {result.solve_time_ms:.2f} ms")
        print(f"  Total cost: {result.cost_total:.4f}")
        print(f"  Reason: {result.reason}")
        
        return True
    except Exception as e:
        print(f"✗ Task-space MPC solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_impedance_controller_import():
    """Test that impedance controller module can be imported."""
    print("\n" + "=" * 70)
    print("TEST: Impedance Controller Import")
    print("=" * 70)
    
    try:
        from rfsn.impedance_controller import (
            ImpedanceController,
            ImpedanceConfig,
            ImpedanceProfiles
        )
        print("✓ Impedance controller module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import impedance controller: {e}")
        return False


def test_impedance_config():
    """Test impedance configuration."""
    print("\n" + "=" * 70)
    print("TEST: Impedance Configuration")
    print("=" * 70)
    
    try:
        from rfsn.impedance_controller import ImpedanceConfig
        
        # Test default config
        config = ImpedanceConfig()
        assert config.K_pos is not None
        assert config.K_ori is not None
        assert config.D_pos is not None
        assert config.D_ori is not None
        assert len(config.K_pos) == 3
        assert len(config.K_ori) == 3
        
        print("✓ Impedance config created with defaults")
        print(f"  K_pos: {config.K_pos}")
        print(f"  K_ori: {config.K_ori}")
        
        # Test custom config
        config_custom = ImpedanceConfig(
            K_pos=np.array([150.0, 150.0, 150.0]),
            K_ori=np.array([15.0, 15.0, 15.0]),
            max_force=40.0
        )
        assert config_custom.max_force == 40.0
        
        print("✓ Impedance config created with custom parameters")
        return True
    except Exception as e:
        print(f"✗ Failed to create impedance config: {e}")
        return False


def test_impedance_profiles():
    """Test pre-tuned impedance profiles."""
    print("\n" + "=" * 70)
    print("TEST: Impedance Profiles")
    print("=" * 70)
    
    try:
        from rfsn.impedance_controller import ImpedanceProfiles
        
        # Test all profiles
        grasp_soft = ImpedanceProfiles.grasp_soft()
        assert grasp_soft.max_force == 30.0
        print("✓ Grasp soft profile created")
        
        grasp_firm = ImpedanceProfiles.grasp_firm()
        assert grasp_firm.max_force == 50.0
        print("✓ Grasp firm profile created")
        
        place_gentle = ImpedanceProfiles.place_gentle()
        assert place_gentle.max_force == 20.0
        print("✓ Place gentle profile created")
        
        transport_stable = ImpedanceProfiles.transport_stable()
        assert transport_stable.max_force == 50.0
        print("✓ Transport stable profile created")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create impedance profiles: {e}")
        return False


def test_impedance_controller_compute():
    """Test impedance controller torque computation."""
    print("\n" + "=" * 70)
    print("TEST: Impedance Controller Compute")
    print("=" * 70)
    
    try:
        from rfsn.impedance_controller import ImpedanceController, ImpedanceConfig
        
        # Load model
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize
        mj.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        mj.mj_forward(model, data)
        
        # Create controller
        config = ImpedanceConfig()
        controller = ImpedanceController(model, config)
        print("✓ Impedance controller created")
        
        # Compute torques
        x_target_pos = np.array([0.4, 0.0, 0.5])
        x_target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        tau = controller.compute_torques(data, x_target_pos, x_target_quat)
        
        # Check output
        assert tau is not None
        assert tau.shape == (7,)
        assert np.all(np.abs(tau) <= 87.0)  # Within torque limits
        
        print(f"✓ Impedance controller computed torques")
        print(f"  Torque range: [{tau.min():.2f}, {tau.max():.2f}] Nm")
        
        return True
    except Exception as e:
        print(f"✗ Impedance controller compute failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_harness_task_space_mpc_mode():
    """Test harness initialization with TASK_SPACE_MPC mode."""
    print("\n" + "=" * 70)
    print("TEST: Harness Task-Space MPC Mode")
    print("=" * 70)
    
    try:
        from rfsn.harness import RFSNHarness
        
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize harness in task-space MPC mode
        harness = RFSNHarness(
            model, data,
            mode="rfsn",
            controller_mode="TASK_SPACE_MPC"
        )
        
        assert harness.task_space_mpc_enabled == True
        assert harness.task_space_solver is not None
        
        print("✓ Harness initialized with TASK_SPACE_MPC mode")
        print(f"  Task-space solver: {harness.task_space_solver}")
        
        return True
    except Exception as e:
        print(f"✗ Harness task-space MPC mode initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_harness_impedance_mode():
    """Test harness initialization with IMPEDANCE mode."""
    print("\n" + "=" * 70)
    print("TEST: Harness Impedance Mode")
    print("=" * 70)
    
    try:
        from rfsn.harness import RFSNHarness
        
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize harness in impedance mode
        harness = RFSNHarness(
            model, data,
            mode="rfsn",
            controller_mode="IMPEDANCE"
        )
        
        assert harness.impedance_enabled == True
        assert harness.impedance_controller is not None
        
        print("✓ Harness initialized with IMPEDANCE mode")
        print(f"  Impedance controller: {harness.impedance_controller}")
        
        return True
    except Exception as e:
        print(f"✗ Harness impedance mode initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_space_mpc_integration():
    """Test full integration of task-space MPC with harness."""
    print("\n" + "=" * 70)
    print("TEST: Task-Space MPC Integration")
    print("=" * 70)
    
    try:
        from rfsn.harness import RFSNHarness
        
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize harness
        mj.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        mj.mj_forward(model, data)
        
        harness = RFSNHarness(
            model, data,
            mode="rfsn",
            controller_mode="TASK_SPACE_MPC"
        )
        
        harness.start_episode()
        
        # Run a few steps
        step_count = 5
        for i in range(step_count):
            obs = harness.step()
            assert obs is not None
        
        print(f"✓ Task-space MPC integration test passed")
        print(f"  Steps executed: {step_count}")
        print(f"  Task-space steps used: {harness.task_space_steps_used}")
        
        return True
    except Exception as e:
        print(f"✗ Task-space MPC integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_impedance_integration():
    """Test full integration of impedance control with harness."""
    print("\n" + "=" * 70)
    print("TEST: Impedance Control Integration")
    print("=" * 70)
    
    try:
        from rfsn.harness import RFSNHarness
        
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize harness
        mj.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        mj.mj_forward(model, data)
        
        harness = RFSNHarness(
            model, data,
            mode="rfsn",
            controller_mode="IMPEDANCE"
        )
        
        harness.start_episode()
        
        # Run a few steps
        step_count = 5
        for i in range(step_count):
            obs = harness.step()
            assert obs is not None
        
        print(f"✓ Impedance control integration test passed")
        print(f"  Steps executed: {step_count}")
        print(f"  Controller mode: {obs.controller_mode}")
        
        return True
    except Exception as e:
        print(f"✗ Impedance control integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_v8_tests():
    """Run all v8 upgrade tests."""
    print("\n" + "=" * 70)
    print("RUNNING ALL V8 UPGRADE TESTS")
    print("=" * 70)
    
    tests = [
        ("Task-Space MPC Import", test_task_space_mpc_module_import),
        ("Task-Space MPC Config", test_task_space_mpc_config),
        ("Task-Space MPC Solver", test_task_space_mpc_solver),
        ("Impedance Controller Import", test_impedance_controller_import),
        ("Impedance Config", test_impedance_config),
        ("Impedance Profiles", test_impedance_profiles),
        ("Impedance Controller Compute", test_impedance_controller_compute),
        ("Harness Task-Space MPC Mode", test_harness_task_space_mpc_mode),
        ("Harness Impedance Mode", test_harness_impedance_mode),
        ("Task-Space MPC Integration", test_task_space_mpc_integration),
        ("Impedance Integration", test_impedance_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed//total}%)")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_v8_tests()
    exit(0 if success else 1)
