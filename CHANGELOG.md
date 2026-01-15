# RFSN-ROBOT Changelog

## Version 6.0 - Grasp Validation Hardening (2026-01-15)

### Major Features

**Enhanced Grasp Validation System**
- Implements object-follows-EE attachment proxy using sliding window history
- Adds real-time slip detection via velocity spikes and position drift
- Requires bilateral contact persistence before allowing GRASP→LIFT transitions
- Continuous monitoring during LIFT with automatic recovery on attachment loss

### New Components

**History Tracking (`GraspHistoryBuffer`)**
- 20-step sliding window for object/EE relative state
- Tracks positions, velocities, and contact patterns
- Enables temporal reasoning about grasp quality

**Attachment Proxy**
- Detects when object follows EE motion
- Computes confidence based on:
  - Relative position stability (std < 1.5cm)
  - Relative velocity magnitude (< 5cm/s)
  - Height correlation (> 0.7)

**Slip Detection**
- Velocity spike detection (15cm/s threshold)
- Position drift monitoring (2cm in 5 steps)
- Contact intermittency tracking

**Contact Persistence**
- Bilateral finger contact validation
- Requires 5 of last 10 steps with both fingers
- Prevents premature lifting

### Enhanced State Machine

**GRASP → LIFT Transition (V6 Mode)**
- Requires ALL conditions:
  - Quality ≥ 0.7
  - Bilateral contact
  - Object attached
  - No slip detected
  - Contact persistent
  - Minimum time satisfied

**LIFT State Monitoring**
- Continuous slip/attachment checks
- Immediate RECOVER on failure

**Backward Compatibility**
- V5 tests pass unchanged
- Graceful fallback when history insufficient
- Auto-detects enhanced vs. basic quality dicts

### Evaluation & Logging

**New Metrics**
- `grasp_attempts`: Count of grasp state entries
- `grasp_confirmed`: Successful GRASP→LIFT transitions
- `false_lift_count`: Lifts that fail within 20 steps
- `grasp_confirmation_time_s`: Time to confirm grasp
- `slip_events`: Slip and attachment loss events
- `grasp_success_rate`: confirmed/attempts
- `false_lift_rate`: false_lifts/confirmed

**Enhanced CSV Output**
- All metrics logged per episode
- Compatible with existing analysis tools

**Event Logging**
- `slip_detected` events with context
- `attachment_lost` events during LIFT

### Files Modified

- `rfsn/mujoco_utils.py`: +400 lines (validation infrastructure)
- `rfsn/obs_packet.py`: Added object velocity fields
- `rfsn/harness.py`: +100 lines (history buffer integration)
- `rfsn/state_machine.py`: Enhanced transitions with fallback
- `rfsn/logger.py`: Grasp metrics tracking
- `eval/metrics.py`: Grasp validation metrics computation

### New Files

- `demo_v6_grasp_validation.py`: Interactive demonstration
- `V6_GRASP_VALIDATION_SUMMARY.md`: Complete implementation guide

### Testing

- ✓ All v5 tests pass (backward compatibility verified)
- ✓ Enhanced validation unit tests
- ✓ Integration tests with history buffer
- ✓ Demo script validates all new features

### Expected Impact

- **Reduced false lift rate**: Stricter validation prevents lifting without confirmed grasp
- **Faster failure recovery**: Early slip detection during LIFT
- **Improved success rate**: False lifts are major failure mode; validation addresses this
- **Better diagnostics**: Rich metrics help tune and debug grasp behavior

### Performance

- Minimal overhead: ~0.1ms per step for full validation
- No impact outside GRASP/LIFT states
- O(window_size) computation with 20-step window

### Breaking Changes

None. All changes are additive with backward compatibility.

### Migration Guide

No migration needed. Enhanced validation activates automatically when:
1. Using `mode="rfsn"` or `mode="rfsn_learning"`
2. State machine is in GRASP or LIFT states
3. History buffer has ≥5 observations

To verify v6 features are active, check log output for:
```
[RFSN] GRASP confirmed (enhanced): quality=0.85, attached=True, persistent=True
```

vs. v5 mode:
```
[RFSN] GRASP confirmed (v5 mode): quality=0.85
```

### Demonstration

Run the demo to see v6 features in action:
```bash
python demo_v6_grasp_validation.py
```

### Documentation

- See `V6_GRASP_VALIDATION_SUMMARY.md` for complete implementation details
- All functions have docstrings with parameter descriptions
- Examples provided in demo script

---

## Version 5.0 - Safety Hardening & Profile Learning (Previous)

See `V5_UPGRADE_SUMMARY.md` for details.

Key features:
- SafetyClamp with poison/rollback system
- UCB-based profile learning
- Fail-loud contact parsing
- Comprehensive logging infrastructure

---

## Earlier Versions

See git history and previous documentation files.
