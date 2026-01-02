#!/usr/bin/env python3
"""
Bionic Arm System Demo
=======================

Demonstrates the complete bionic arm system pipeline:
1. Simulated EEG acquisition
2. Signal preprocessing
3. Feature extraction
4. Neural decoding
5. Arm control
6. Visual feedback

Usage:
    python scripts/demo.py
    python scripts/demo.py --duration 30  # Run for 30 seconds
    python scripts/demo.py --headless      # No visualization

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print demo banner."""
    print(
        """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║            🦾 BIONIC ARM SYSTEM DEMONSTRATION 🧠                 ║
║                                                                  ║
║    Brain-Computer Interface Controlled Prosthetic Arm           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    )


def run_bci_demo(duration: float = 10.0) -> None:
    """
    Run BCI pipeline demonstration.

    Args:
        duration: Demo duration in seconds
    """
    from src.bci.acquisition import AcquisitionConfig, SimulatedAcquisition
    from src.bci.decoder import ContinuousDecoder, DecoderConfig, KalmanFilter
    from src.bci.features import (
        BandPowerConfig,
        BandPowerExtractor,
        FeatureConfig,
        FeatureExtractor,
    )
    from src.bci.preprocessing import Preprocessor, PreprocessorConfig

    print("\n" + "=" * 60)
    print("📡 BCI PIPELINE DEMO")
    print("=" * 60)

    # Configure acquisition
    acq_config = AcquisitionConfig(
        sampling_rate=250, n_channels=8, buffer_size_seconds=2.0
    )

    # Configure preprocessing
    prep_config = PreprocessorConfig(
        sampling_rate=250,
        n_channels=8,
    )

    # Configure feature extraction
    bp_config = BandPowerConfig(
        bands={
            "mu": (8.0, 12.0),
            "beta": (12.0, 30.0),
        },
        sampling_rate=250,
    )

    # Configure decoder
    dec_config = DecoderConfig(
        n_features=16,  # 8 channels * 2 bands
        n_outputs=7,
        use_kalman=True,
    )

    print("\n📊 Configuration:")
    print(f"   • Channels: {acq_config.n_channels}")
    print(f"   • Sampling rate: {acq_config.sampling_rate} Hz")
    print(f"   • Buffer: {acq_config.buffer_size_seconds}s")
    print(f"   • Output dimensions: {dec_config.n_outputs}")

    # Create components
    acquisition = SimulatedAcquisition(acq_config)
    preprocessor = Preprocessor(prep_config)
    feature_extractor = BandPowerExtractor(bp_config)
    decoder = ContinuousDecoder(dec_config)

    print("\n✅ Components created successfully")

    # Start acquisition
    acquisition.connect()
    acquisition.start()
    print("✅ Acquisition started")

    # Wait for buffer to fill
    time.sleep(0.5)

    print(f"\n🔄 Running BCI pipeline for {duration}s...")
    print("-" * 60)

    start_time = time.time()
    update_count = 0
    latencies = []

    try:
        while (time.time() - start_time) < duration:
            loop_start = time.perf_counter()

            # Get latest EEG data
            eeg_data = acquisition.get_latest_data(250)  # 1 second

            if eeg_data is not None:
                # Preprocess
                processed = preprocessor.process(eeg_data)

                # Extract features
                features = feature_extractor.extract(processed)

                # Decode to velocity - returns (velocity, confidence) tuple
                result = decoder.decode(features)
                velocity, confidence = result

                update_count += 1
                latency = (time.perf_counter() - loop_start) * 1000
                latencies.append(latency)

                # Print update every 0.5 seconds
                if update_count % 15 == 0:
                    # Ensure velocity is a flat numpy array
                    vel = np.atleast_1d(velocity).flatten()
                    v0 = float(vel[0]) if len(vel) > 0 else 0.0
                    v1 = float(vel[1]) if len(vel) > 1 else 0.0
                    print(
                        f"   Update {update_count:4d} | "
                        f"Latency: {latency:5.1f}ms | "
                        f"Velocity: [{v0:+.3f}, {v1:+.3f}, ...] | "
                        f"Confidence: {float(confidence):.2f}"
                    )

            # Control loop timing (~30 Hz)
            time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")

    # Cleanup
    acquisition.stop()
    acquisition.disconnect()

    # Statistics
    print("-" * 60)
    print("\n📈 Performance Statistics:")
    print(f"   • Total updates: {update_count}")
    print(f"   • Update rate: {update_count / duration:.1f} Hz")
    print(f"   • Mean latency: {np.mean(latencies):.1f} ms")
    print(f"   • Max latency: {np.max(latencies):.1f} ms")
    print(f"   • P95 latency: {np.percentile(latencies, 95):.1f} ms")


def run_control_demo(duration: float = 5.0) -> None:
    """
    Run arm control demonstration.

    Args:
        duration: Demo duration in seconds
    """
    from src.control.controller import ArmController, ControllerConfig, ControlMode
    from src.control.kinematics import ArmKinematics, Transform
    from src.control.trajectory import TrajectoryConfig, TrajectoryGenerator

    print("\n" + "=" * 60)
    print("🤖 ARM CONTROL DEMO")
    print("=" * 60)

    # Create arm kinematics
    arm = ArmKinematics.default_arm()
    print("\n✅ Arm kinematics created (7 DOF)")

    # Forward kinematics demo
    print("\n📐 Forward Kinematics:")
    q_home = np.zeros(7)
    ee_home = arm.forward_kinematics(q_home)
    print(f"   Home position: {ee_home.position}")

    q_test = np.array([0.3, 0.2, 0.0, -0.5, 0.0, 0.3, 0.0])
    ee_test = arm.forward_kinematics(q_test)
    print(f"   Test position: {ee_test.position}")

    # Jacobian
    J = arm.jacobian(q_home)
    print(f"\n📊 Jacobian shape: {J.shape}")
    print(f"   Condition number: {np.linalg.cond(J):.2f}")

    # Inverse kinematics demo
    print("\n🎯 Inverse Kinematics:")
    target = arm.forward_kinematics(q_test)  # Use known reachable target
    q_solution, success = arm.inverse_kinematics(target, q_home)

    if success:
        achieved = arm.forward_kinematics(q_solution)
        error = np.linalg.norm(target.position - achieved.position)
        print(f"   ✅ IK successful! Position error: {error*1000:.2f} mm")
    else:
        print("   ❌ IK failed")

    # Trajectory generation
    print("\n📈 Trajectory Generation:")
    traj_config = TrajectoryConfig(
        max_velocity=1.0,
        max_acceleration=5.0,
    )
    traj_gen = TrajectoryGenerator(traj_config)

    segment = traj_gen.point_to_point(q_home, q_test, duration=2.0)
    print(f"   Generated {len(segment.points)} waypoints")
    print(f"   Duration: {segment.duration:.2f}s")

    # Controller demo
    print("\n🎮 Controller Demo:")
    ctrl_config = ControllerConfig(n_joints=7, control_rate_hz=100.0)
    controller = ArmController(ctrl_config)

    controller.enable()
    print("   ✅ Controller enabled")

    controller.set_control_mode(ControlMode.POSITION)
    controller.set_position_command(q_test)

    # Simulate control loop
    print(f"\n🔄 Running control loop for {duration}s...")

    start_time = time.time()
    while (time.time() - start_time) < duration:
        controller.update()
        time.sleep(0.01)  # 100 Hz

    current = controller.current_position
    print(f"   Final position error: {np.linalg.norm(current - q_test):.4f} rad")

    controller.disable()
    print("   ✅ Controller disabled")


def run_simulation_demo(duration: float = 3.0) -> None:
    """
    Run arm simulation demonstration.

    Args:
        duration: Demo duration in seconds
    """
    from src.simulation.arm_simulator import (
        ArmSimulator,
        PhysicsBackend,
        SimulatorConfig,
    )

    print("\n" + "=" * 60)
    print("🎮 ARM SIMULATION DEMO")
    print("=" * 60)

    # Create simulator with simple physics (no PyBullet required)
    config = SimulatorConfig(
        backend=PhysicsBackend.SIMPLE,
        time_step=0.001,
    )

    sim = ArmSimulator(config)
    print("\n✅ Simulator created (Simple Physics backend)")

    # Initialize
    sim.initialize()
    print("✅ Simulator initialized")

    # Set target position
    target = np.array([0.5, 0.3, 0.0, -0.5, 0.2, 0.1, 0.0])
    sim.set_position_target(target)
    print(f"\n🎯 Target position: {target}")

    # Run simulation
    print(f"\n🔄 Running simulation for {duration}s...")

    sim_steps = int(duration / config.time_step)
    for i in range(sim_steps):
        sim.step()

        if i % 1000 == 0:
            state = sim.get_state()
            pos_error = np.linalg.norm(state.joint_positions - target)
            print(f"   Step {i:6d} | Position error: {pos_error:.4f} rad")

    # Final state
    final_state = sim.get_state()
    print(f"\n📊 Final state:")
    print(f"   Positions: {np.round(final_state.joint_positions, 3)}")
    print(
        f"   Position error: {np.linalg.norm(final_state.joint_positions - target):.4f} rad"
    )

    sim.cleanup()
    print("✅ Simulator cleaned up")


def run_grasping_demo() -> None:
    """Run grasping demonstration."""
    from src.control.grasping import GraspConfig, GraspController, GraspPhase, GraspType

    print("\n" + "=" * 60)
    print("🤏 GRASPING DEMO")
    print("=" * 60)

    config = GraspConfig(
        n_fingers=5,
        force_min=0.5,
        force_max=30.0,
    )

    # Test different grasps - create fresh controller for each to avoid state issues
    grasp_types = [GraspType.POWER, GraspType.PRECISION, GraspType.LATERAL]

    for grasp_type in grasp_types:
        # Fresh controller for each grasp test
        controller = GraspController(config)

        print(f"\n🤚 Testing {grasp_type.name} grasp:")

        # Initiate the grasp
        success = controller.initiate_grasp(grasp_type)
        if not success:
            print(f"   ❌ Failed to initiate grasp")
            continue

        primitive = controller.primitives[grasp_type]
        print(f"   Grasp type: {primitive.grasp_type.name}")
        print(f"   Preshape: {np.round(primitive.preshape[:4], 2)}...")
        print(f"   Final shape: {np.round(primitive.final_shape[:4], 2)}...")

        # Simulate grasp phases with realistic timing
        positions = np.zeros(10)
        sim_time = 0.0
        dt = 0.01

        # Run through preshape, approach, close until we reach HOLD
        for i in range(100):  # Up to 1 second
            sim_time += dt
            force = min(3.0 + i * 0.1, 10.0)  # Increasing force simulates contact
            positions = controller.update(
                current_positions=positions,
                current_forces=np.ones(5) * force,
                dt=dt,
                time=sim_time,
            )
            if controller.phase == GraspPhase.HOLD:
                break

        print(f"   Reached phase: {controller.phase.name}")

        # Release the grasp
        controller.release()

        # Run release updates until back to IDLE
        for i in range(50):  # 0.5 seconds for release
            sim_time += dt
            positions = controller.update(
                current_positions=positions,
                current_forces=np.ones(5) * 0.5,
                dt=dt,
                time=sim_time,
            )
            if controller.phase == GraspPhase.IDLE:
                break

        print(f"   ✅ Grasp completed, final phase: {controller.phase.name}")

    print("\n✅ Grasp controller tested successfully")


def run_kalman_demo() -> None:
    """Demonstrate Kalman filter smoothing."""
    from src.bci.decoder import KalmanFilter

    print("\n" + "=" * 60)
    print("📊 KALMAN FILTER DEMO")
    print("=" * 60)

    # Create Kalman filter
    kf = KalmanFilter(n_dims=1, process_noise=0.001, measurement_noise=0.1)
    print("\n✅ Kalman filter created")
    print(f"   Process noise: 0.001")
    print(f"   Measurement noise: 0.1")

    # Generate noisy sinusoidal signal
    t = np.linspace(0, 4 * np.pi, 200)
    true_signal = np.sin(t)
    noisy_signal = true_signal + np.random.randn(len(t)) * 0.3

    # Filter
    filtered_signal = []
    for z in noisy_signal:
        estimate = kf.filter(np.array([z]))
        filtered_signal.append(estimate[0])

    filtered_signal = np.array(filtered_signal)

    # Statistics
    noisy_rmse = np.sqrt(np.mean((noisy_signal - true_signal) ** 2))
    filtered_rmse = np.sqrt(np.mean((filtered_signal - true_signal) ** 2))

    print(f"\n📈 Results:")
    print(f"   Noisy signal RMSE: {noisy_rmse:.4f}")
    print(f"   Filtered signal RMSE: {filtered_rmse:.4f}")
    print(f"   Improvement: {(1 - filtered_rmse/noisy_rmse)*100:.1f}%")


def run_full_demo(duration: float = 10.0) -> None:
    """Run full system demonstration."""
    print_banner()

    print("This demo showcases the bionic arm system components:")
    print("  • BCI Pipeline: EEG acquisition → preprocessing → decoding")
    print("  • Arm Control: Kinematics, trajectory planning, motor control")
    print("  • Simulation: Physics-based arm model")
    print("  • Grasping: Multiple grasp primitives and force control")

    # Run each demo
    try:
        run_bci_demo(duration / 2)
        run_kalman_demo()
        run_control_demo(duration / 4)
        run_simulation_demo(duration / 4)
        run_grasping_demo()

        print("\n" + "=" * 60)
        print("✨ DEMO COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("  • Read docs/project-plan.md for the development roadmap")
        print("  • Check src/ for implementation details")
        print("  • Run tests with: pytest tests/ -v")
        print("  • See README.md for project overview")

    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bionic Arm System Demonstration")
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="Demo duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--bci-only", action="store_true", help="Run only BCI pipeline demo"
    )
    parser.add_argument(
        "--control-only", action="store_true", help="Run only control system demo"
    )
    parser.add_argument(
        "--sim-only", action="store_true", help="Run only simulation demo"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.bci_only:
        print_banner()
        run_bci_demo(args.duration)
    elif args.control_only:
        print_banner()
        run_control_demo(args.duration)
    elif args.sim_only:
        print_banner()
        run_simulation_demo(args.duration)
    else:
        run_full_demo(args.duration)


if __name__ == "__main__":
    main()
