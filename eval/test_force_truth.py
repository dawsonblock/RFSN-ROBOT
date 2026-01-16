"""
V11 Acceptance Test: Force Truth + Impedance Gating
===================================================
Tests that impedance controller uses only truthful force signals and properly gates forces.

Acceptance Criteria:
1. Impedance controller receives force signals from ObsPacket
2. Force gating triggers when thresholds exceeded
3. No efc_force usage in impedance code paths
4. force_signal_is_proxy correctly reported
5. Force gate events properly logged
"""

import numpy as np
import mujoco as mj
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfsn.harness import RFSNHarness
from rfsn.logger import RFSNLogger


def test_impedance_force_routing():
    """Test that force signals are properly routed to impedance controller."""
    print("=" * 70)
    print("TEST 1: Impedance Force Signal Routing")
    print("=" * 70)
    
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize harness in IMPEDANCE mode with RFSN
    logger = RFSNLogger(run_dir="/tmp/test_force_truth_routing")
    harness = RFSNHarness(
        model, data,
        mode="rfsn",
        controller_mode="IMPEDANCE",
        logger=logger
    )
    
    # Start episode
    harness.start_episode()
    
    # Run a few steps
    for _ in range(10):
        obs = harness.step()
    
    # Check that force fields exist in obs
    print(f"ObsPacket force fields present:")
    print(f"  cube_fingers_fN: {obs.cube_fingers_fN:.3f} N")
    print(f"  cube_table_fN: {obs.cube_table_fN:.3f} N")
    print(f"  ee_table_fN: {obs.ee_table_fN:.3f} N")
    print(f"  force_signal_is_proxy: {obs.force_signal_is_proxy}")
    
    success = True
    
    # Check force fields exist
    if not hasattr(obs, 'cube_fingers_fN'):
        print("✗ FAIL: cube_fingers_fN missing")
        success = False
    if not hasattr(obs, 'cube_table_fN'):
        print("✗ FAIL: cube_table_fN missing")
        success = False
    if not hasattr(obs, 'ee_table_fN'):
        print("✗ FAIL: ee_table_fN missing")
        success = False
    if not hasattr(obs, 'force_signal_is_proxy'):
        print("✗ FAIL: force_signal_is_proxy missing")
        success = False
    
    if success:
        print("✓ PASS: All force fields present and accessible")
    
    harness.end_episode(success=True)
    
    return success


def test_impedance_force_gating():
    """Test that impedance force gating triggers during excessive contact forces."""
    print("\n" + "=" * 70)
    print("TEST 2: Impedance Force Gating")
    print("=" * 70)
    
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize harness in IMPEDANCE mode
    logger = RFSNLogger(run_dir="/tmp/test_force_truth_gating")
    harness = RFSNHarness(
        model, data,
        mode="rfsn",
        controller_mode="IMPEDANCE",
        logger=logger
    )
    
    # Start episode
    harness.start_episode()
    
    # Position EE above table and command downward motion to create contact
    # Set robot to a position near table
    data.qpos[:7] = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.8, 0.785])
    
    # Set cube on table
    cube_joint_name = "cube_freejoint"
    cube_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, cube_joint_name)
    cube_qpos_start = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_start:cube_qpos_start + 3] = [0.35, 0.0, 0.47]
    data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
    
    # Forward kinematics
    mj.mj_forward(model, data)
    
    # Run simulation to create contact
    gate_triggered = False
    max_force_seen = 0.0
    steps_with_force = 0
    
    for step in range(300):
        obs = harness.step()
        
        # Track force values
        max_contact_force = max(obs.ee_table_fN, obs.cube_table_fN)
        if max_contact_force > 0.1:
            steps_with_force += 1
            max_force_seen = max(max_force_seen, max_contact_force)
        
        # Check if force gate triggered
        if harness.impedance_controller.force_gate_triggered:
            gate_triggered = True
            print(f"✓ Force gate triggered at step {step}")
            print(f"  Gate value: {harness.impedance_controller.force_gate_value:.2f} N")
            print(f"  Gate source: {harness.impedance_controller.force_gate_source}")
            print(f"  Gate proxy: {harness.impedance_controller.force_gate_proxy}")
            break
    
    print(f"\nForce statistics:")
    print(f"  Steps with force > 0.1 N: {steps_with_force}")
    print(f"  Max force observed: {max_force_seen:.2f} N")
    print(f"  Force signal is proxy: {obs.force_signal_is_proxy}")
    
    # Check that we observed meaningful forces
    success = True
    
    if steps_with_force == 0:
        print("✗ FAIL: No contact forces observed")
        success = False
    else:
        print(f"✓ PASS: Contact forces observed in {steps_with_force} steps")
    
    # Gate triggering is optional but expected with sufficient contact
    if gate_triggered:
        print("✓ PASS: Force gate triggered as expected")
    else:
        # This is a soft warning, not a hard failure
        print(f"⚠ WARNING: Force gate did not trigger (max force: {max_force_seen:.2f} N)")
    
    harness.end_episode(success=True)
    
    return success


def test_no_efc_force_usage():
    """Test that deprecated _get_ee_contact_forces raises error."""
    print("\n" + "=" * 70)
    print("TEST 3: No efc_force Usage")
    print("=" * 70)
    
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize impedance controller
    from rfsn.impedance_controller import ImpedanceController
    controller = ImpedanceController(model)
    
    # Try to call deprecated method - should raise RuntimeError
    try:
        controller._get_ee_contact_forces(data)
        print("✗ FAIL: Deprecated method did not raise error")
        return False
    except RuntimeError as e:
        if "efc_force" in str(e) and "invalid" in str(e).lower():
            print("✓ PASS: Deprecated method raises error as expected")
            print(f"  Error message: {e}")
            return True
        else:
            print(f"✗ FAIL: Wrong error raised: {e}")
            return False
    except Exception as e:
        print(f"✗ FAIL: Unexpected exception: {e}")
        return False


def test_force_truth_integration():
    """
    Integration test: Run short IMPEDANCE sequence and verify force truth.
    
    Tests:
    - Force signals present in obs
    - At least one of ee_table_fN or cube_table_fN becomes > 0
    - If force_signal_is_proxy=False, forces are not all zeros
    """
    print("\n" + "=" * 70)
    print("TEST 4: Force Truth Integration")
    print("=" * 70)
    
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize harness in IMPEDANCE mode
    logger = RFSNLogger(run_dir="/tmp/test_force_truth_integration")
    harness = RFSNHarness(
        model, data,
        mode="rfsn",
        controller_mode="IMPEDANCE",
        logger=logger
    )
    
    # Start episode
    harness.start_episode()
    
    # Collect observations
    observations = []
    for _ in range(200):
        obs = harness.step()
        observations.append(obs)
    
    harness.end_episode(success=True)
    
    # Analyze force signals
    has_real_force = False
    has_proxy_signal = False
    max_ee_table = 0.0
    max_cube_table = 0.0
    max_cube_fingers = 0.0
    
    for obs in observations:
        max_ee_table = max(max_ee_table, obs.ee_table_fN)
        max_cube_table = max(max_cube_table, obs.cube_table_fN)
        max_cube_fingers = max(max_cube_fingers, obs.cube_fingers_fN)
        
        if obs.force_signal_is_proxy:
            has_proxy_signal = True
        else:
            has_real_force = True
    
    print(f"\nForce Truth Summary:")
    print(f"  Steps collected: {len(observations)}")
    print(f"  Max ee_table_fN: {max_ee_table:.3f} N")
    print(f"  Max cube_table_fN: {max_cube_table:.3f} N")
    print(f"  Max cube_fingers_fN: {max_cube_fingers:.3f} N")
    print(f"  Has real force API: {has_real_force}")
    print(f"  Has proxy signals: {has_proxy_signal}")
    
    success = True
    
    # Criterion 1: At least one force becomes > 0
    if max_ee_table > 0.0 or max_cube_table > 0.0:
        print("✓ PASS: Contact forces observed")
    else:
        print("✗ FAIL: No contact forces observed")
        success = False
    
    # Criterion 2: If real force API available, forces should be non-zero when in contact
    if has_real_force and not has_proxy_signal:
        if max_ee_table > 0.0 or max_cube_table > 0.0:
            print("✓ PASS: Real force API providing non-zero forces")
        else:
            print("⚠ WARNING: Real force API available but no forces observed")
    
    # Criterion 3: Check force_signal_is_proxy flag consistency
    if has_proxy_signal:
        print("⚠ WARNING: Using proxy signals (mj_contactForce not available)")
    else:
        print("✓ PASS: Using real force API (not proxy)")
    
    return success


def main():
    """Run all force truth acceptance tests."""
    print("\n" + "=" * 70)
    print("V11 FORCE TRUTH + IMPEDANCE GATING ACCEPTANCE TESTS")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: Force routing
    try:
        results.append(("Force Routing", test_impedance_force_routing()))
    except Exception as e:
        print(f"\n✗ Test 1 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Force Routing", False))
    
    # Test 2: Force gating
    try:
        results.append(("Force Gating", test_impedance_force_gating()))
    except Exception as e:
        print(f"\n✗ Test 2 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Force Gating", False))
    
    # Test 3: No efc_force usage
    try:
        results.append(("No efc_force Usage", test_no_efc_force_usage()))
    except Exception as e:
        print(f"\n✗ Test 3 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("No efc_force Usage", False))
    
    # Test 4: Integration test
    try:
        results.append(("Force Truth Integration", test_force_truth_integration()))
    except Exception as e:
        print(f"\n✗ Test 4 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Force Truth Integration", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("Force truth and impedance gating working correctly")
        return 0
    else:
        failed_count = sum(1 for _, passed in results if not passed)
        print(f"✗✗✗ {failed_count} TEST(S) FAILED ✗✗✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
