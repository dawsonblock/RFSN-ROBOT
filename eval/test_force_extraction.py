"""
V10 Acceptance Test: Force Extraction Validation
================================================
Tests that contact force signals are properly extracted and trigger gating.

Acceptance Criteria:
1. With forced downward motion in PLACE, cube_table_fN rises meaningfully
2. Force gate triggers when threshold exceeded
3. force_signal_is_proxy=False when real forces available
"""

import numpy as np
import mujoco as mj
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfsn.mujoco_utils import (
    init_id_cache, compute_contact_wrenches, build_obs_packet
)


def test_force_extraction_basic():
    """Test basic force extraction functionality."""
    print("=" * 70)
    print("TEST 1: Basic Force Extraction")
    print("=" * 70)
    
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize ID cache
    init_id_cache(model)
    
    # Reset to default pose with cube on table
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    
    # Set cube on table
    cube_joint_name = "cube_freejoint"
    cube_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, cube_joint_name)
    cube_qpos_start = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_start:cube_qpos_start + 3] = [0.3, 0.0, 0.47]
    data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
    
    # Zero velocities
    data.qvel[:] = 0.0
    
    # Forward simulation to settle
    for _ in range(100):
        mj.mj_step(model, data)
    
    # Extract contact wrenches
    wrenches = compute_contact_wrenches(model, data)
    
    print(f"Force signal is proxy: {wrenches['force_signal_is_proxy']}")
    print(f"Cube-table force: {np.linalg.norm(wrenches['cube_table_force_world']):.2f} N")
    print(f"Cube-fingers force: {np.linalg.norm(wrenches['cube_fingers_force_world']):.2f} N")
    print(f"EE-table force: {np.linalg.norm(wrenches['ee_table_force_world']):.2f} N")
    print(f"Number of contacts: {len(wrenches['contacts'])}")
    
    # Check acceptance criteria
    success = True
    
    # Criterion 1: Real force API should be used (not proxy)
    if wrenches['force_signal_is_proxy']:
        print("\n✗ FAIL: Using force proxy instead of real API")
        success = False
    else:
        print("\n✓ PASS: Using real force API (mj.mj_contactForce)")
    
    # Criterion 2: Cube should have meaningful contact force with table (gravity)
    cube_table_fN = np.linalg.norm(wrenches['cube_table_force_world'])
    # Cube mass ~0.05kg, so expect ~0.5N from gravity
    if cube_table_fN < 0.1:
        print(f"✗ FAIL: Cube-table force too small ({cube_table_fN:.3f} N)")
        success = False
    else:
        print(f"✓ PASS: Cube-table force is meaningful ({cube_table_fN:.3f} N)")
    
    return success


def test_force_gating_trigger():
    """Test that force gating triggers during excessive downward force."""
    print("\n" + "=" * 70)
    print("TEST 2: Force Gating Trigger")
    print("=" * 70)
    
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize ID cache
    init_id_cache(model)
    
    # Reset to pose where EE is above table
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    
    # Set cube on table
    cube_joint_name = "cube_freejoint"
    cube_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, cube_joint_name)
    cube_qpos_start = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_start:cube_qpos_start + 3] = [0.3, 0.0, 0.47]
    data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
    
    # Zero velocities
    data.qvel[:] = 0.0
    
    # Forward to settle
    for _ in range(50):
        mj.mj_step(model, data)
    
    # Apply excessive downward force on cube
    # Get cube body ID
    cube_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "cube")
    
    # Apply downward external force
    force_threshold = 15.0  # Same as harness threshold
    applied_force = force_threshold * 2.0  # Apply 2x threshold
    
    for step in range(100):
        # Apply downward force to cube
        data.xfrc_applied[cube_body_id, 2] = -applied_force  # Negative Z is down
        
        mj.mj_step(model, data)
        
        # Check force after some settling
        if step > 50:
            wrenches = compute_contact_wrenches(model, data)
            cube_table_fN = np.linalg.norm(wrenches['cube_table_force_world'])
            
            if cube_table_fN > force_threshold:
                print(f"✓ PASS: Force gate would trigger at step {step}")
                print(f"  cube_table_fN = {cube_table_fN:.2f} N > threshold {force_threshold:.2f} N")
                return True
    
    print(f"✗ FAIL: Force never exceeded threshold")
    return False


def test_obs_packet_integration():
    """Test that force signals are properly integrated into ObsPacket."""
    print("\n" + "=" * 70)
    print("TEST 3: ObsPacket Integration")
    print("=" * 70)
    
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize ID cache
    init_id_cache(model)
    
    # Reset
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    
    # Set cube on table
    cube_joint_name = "cube_freejoint"
    cube_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, cube_joint_name)
    cube_qpos_start = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_start:cube_qpos_start + 3] = [0.3, 0.0, 0.47]
    data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
    
    # Zero velocities
    data.qvel[:] = 0.0
    
    # Forward to settle
    for _ in range(100):
        mj.mj_step(model, data)
    
    # Build obs packet
    obs = build_obs_packet(model, data, t=0.0, dt=0.002)
    
    # Check that force fields exist and are populated
    print(f"ObsPacket force fields:")
    print(f"  cube_fingers_fN: {obs.cube_fingers_fN:.3f} N")
    print(f"  cube_table_fN: {obs.cube_table_fN:.3f} N")
    print(f"  ee_table_fN: {obs.ee_table_fN:.3f} N")
    print(f"  force_signal_is_proxy: {obs.force_signal_is_proxy}")
    
    # Check that fields are accessible
    success = True
    if not hasattr(obs, 'cube_fingers_fN'):
        print("✗ FAIL: cube_fingers_fN missing from ObsPacket")
        success = False
    if not hasattr(obs, 'cube_table_fN'):
        print("✗ FAIL: cube_table_fN missing from ObsPacket")
        success = False
    if not hasattr(obs, 'ee_table_fN'):
        print("✗ FAIL: ee_table_fN missing from ObsPacket")
        success = False
    if not hasattr(obs, 'force_signal_is_proxy'):
        print("✗ FAIL: force_signal_is_proxy missing from ObsPacket")
        success = False
    
    if success:
        print("✓ PASS: All force fields present in ObsPacket")
    
    # Check that cube_table force is reasonable
    if obs.cube_table_fN < 0.1:
        print(f"✗ FAIL: cube_table_fN too small ({obs.cube_table_fN:.3f} N)")
        success = False
    else:
        print(f"✓ PASS: cube_table_fN is reasonable ({obs.cube_table_fN:.3f} N)")
    
    return success


def main():
    """Run all force extraction tests."""
    print("\n" + "=" * 70)
    print("V10 FORCE EXTRACTION ACCEPTANCE TESTS")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: Basic force extraction
    try:
        results.append(("Basic Force Extraction", test_force_extraction_basic()))
    except Exception as e:
        print(f"\n✗ Test 1 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Basic Force Extraction", False))
    
    # Test 2: Force gating trigger
    try:
        results.append(("Force Gating Trigger", test_force_gating_trigger()))
    except Exception as e:
        print(f"\n✗ Test 2 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Force Gating Trigger", False))
    
    # Test 3: ObsPacket integration
    try:
        results.append(("ObsPacket Integration", test_obs_packet_integration()))
    except Exception as e:
        print(f"\n✗ Test 3 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ObsPacket Integration", False))
    
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
        print("Force extraction is working correctly")
        return 0
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
