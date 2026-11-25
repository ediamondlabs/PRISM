#!/usr/bin/env python3
"""
SUMO Data Collector with Simulated GPS/IMU Sensors
Generates realistic trajectory data with sensor noise
"""

import traci
import pandas as pd
import numpy as np
import os
from datetime import datetime

class SensorSimulator:
    """Simulates GPS and IMU sensor readings with realistic noise"""
    
    def __init__(self):
        # GPS noise parameters (meters)
        self.gps_noise_std = 2.0  # Standard GPS accuracy ~2m
        
        # IMU noise parameters
        self.accel_noise_std = 0.1  # m/s²
        self.gyro_noise_std = 0.01  # rad/s
        
        # Reference point for lat/lon conversion (Barcelona coordinates)
        self.lat_origin = 41.3851
        self.lon_origin = 2.1734
        
    def xy_to_latlon(self, x, y):
        """Convert SUMO XY coordinates to GPS lat/lon"""
        # Approximate conversion (1 degree ≈ 111km)
        lat = self.lat_origin + (y / 111000.0) + np.random.normal(0, self.gps_noise_std/111000.0)
        lon = self.lon_origin + (x / (111000.0 * np.cos(np.radians(self.lat_origin)))) + \
              np.random.normal(0, self.gps_noise_std/111000.0)
        return lat, lon
    
    def compute_acceleration(self, speed_current, speed_prev, dt=0.1):
        """Compute acceleration with IMU noise"""
        accel = (speed_current - speed_prev) / dt
        return accel + np.random.normal(0, self.accel_noise_std)
    
    def compute_gyroscope(self, angle_current, angle_prev, dt=0.1):
        """Compute angular velocity with gyroscope noise"""
        angle_diff = (angle_current - angle_prev) % 360
        if angle_diff > 180:
            angle_diff -= 360
        gyro_z = np.radians(angle_diff) / dt
        return gyro_z + np.random.normal(0, self.gyro_noise_std)

class SUMODataCollector:
    def __init__(self, sumo_cfg):
        self.sumo_cfg = sumo_cfg
        self.sensor_sim = SensorSimulator()
        self.data_buffer = []
        self.vehicle_history = {}  # Store previous states
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = '../data/raw'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def start_simulation(self):
        """Start SUMO with traffic"""
        print("🚀 Starting SUMO simulation...")
        
        sumo_cmd = [
            "sumo",
            "-c", self.sumo_cfg,
            "--step-length", "0.1",
            "--time-to-teleport", "300",
            "--collision.action", "warn",
            "--random"
        ]
        
        traci.start(sumo_cmd)
        print("✅ SUMO connected")
        
    def collect_frame_data(self, step):
        """Collect data with simulated sensors for current frame"""
        timestamp = step * 0.1
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                # Get vehicle state from SUMO
                pos = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                
                # Get previous state
                if veh_id not in self.vehicle_history:
                    self.vehicle_history[veh_id] = {
                        'speed': speed,
                        'angle': angle,
                        'x': pos[0],
                        'y': pos[1]
                    }
                    continue  # Skip first frame (need 2 frames for derivatives)
                
                prev_state = self.vehicle_history[veh_id]
                
                # Simulate GPS
                lat, lon = self.sensor_sim.xy_to_latlon(pos[0], pos[1])
                altitude = 50.0 + np.random.normal(0, 1.0)  # Simulated altitude
                
                # Simulate IMU - Accelerometer
                accel_longitudinal = self.sensor_sim.compute_acceleration(speed, prev_state['speed'])
                accel_x = accel_longitudinal * np.cos(np.radians(angle))
                accel_y = accel_longitudinal * np.sin(np.radians(angle))
                accel_z = -9.81 + np.random.normal(0, 0.05)  # Gravity + noise
                
                # Simulate IMU - Gyroscope
                gyro_x = np.random.normal(0, 0.005)  # Pitch rate
                gyro_y = np.random.normal(0, 0.005)  # Roll rate
                gyro_z = self.sensor_sim.compute_gyroscope(angle, prev_state['angle'])
                
                # Compass (magnetic heading)
                compass = np.radians(angle) + np.random.normal(0, 0.01)
                
                # Orientation (roll, pitch, yaw)
                roll = np.random.normal(0, 0.5)   # Small roll variations
                pitch = np.random.normal(0, 0.5)  # Small pitch variations
                yaw = angle
                
                # Velocity components
                velocity_x = speed * np.cos(np.radians(angle))
                velocity_y = speed * np.sin(np.radians(angle))
                velocity_z = 0.0
                
                # Create record matching CARLA format
                record = {
                    'frame': step,
                    'timestamp': timestamp,
                    'vehicle_id': veh_id,
                    
                    # GPS
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': altitude,
                    
                    # Position (world coordinates)
                    'x': pos[0],
                    'y': pos[1],
                    'z': 0.3,  # Simulated ground height
                    
                    # Orientation
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    
                    # Velocity
                    'velocity_x': velocity_x,
                    'velocity_y': velocity_y,
                    'velocity_z': velocity_z,
                    'speed': speed,
                    
                    # IMU - Accelerometer
                    'accel_x': accel_x,
                    'accel_y': accel_y,
                    'accel_z': accel_z,
                    
                    # IMU - Gyroscope
                    'gyro_x': gyro_x,
                    'gyro_y': gyro_y,
                    'gyro_z': gyro_z,
                    
                    # Compass
                    'compass': compass,
                }
                
                self.data_buffer.append(record)
                
                # Update history
                self.vehicle_history[veh_id] = {
                    'speed': speed,
                    'angle': angle,
                    'x': pos[0],
                    'y': pos[1]
                }
                
            except traci.exceptions.TraCIException:
                pass
    
    def run_collection(self, duration_seconds=3600):
        """Main collection loop"""
        total_steps = int(duration_seconds / 0.1)
        print(f"📊 Collecting data for {duration_seconds}s ({total_steps} steps)...")
        print("⏱️ Update rate: 10 Hz")
        print("-" * 60)
        
        for step in range(total_steps):
            traci.simulationStep()
            self.collect_frame_data(step)
            
            if step % 1000 == 0:  # Progress every 100 seconds
                vehicles = traci.vehicle.getIDList()
                print(f"⏱️ Step {step}/{total_steps} | "
                      f"Time: {step*0.1:.1f}s | "
                      f"Vehicles: {len(vehicles)} | "
                      f"Records: {len(self.data_buffer)}")
        
        print("\n✅ Collection complete!")
        
    def save_data(self):
        """Save dataset to CSV"""
        df = pd.DataFrame(self.data_buffer)
        output_file = f"{self.output_dir}/v2v_dataset_{self.session_id}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n📁 Dataset saved: {output_file}")
        print("📊 Statistics:")
        print(f"  - Total records: {len(df):,}")
        print(f"  - Unique vehicles: {df['vehicle_id'].nunique()}")
        print(f"  - Duration: {df['timestamp'].max():.1f}s")
        print(f"  - Avg records/vehicle: {len(df)/df['vehicle_id'].nunique():.0f}")
        
        # Save metadata
        metadata = {
            'session_id': self.session_id,
            'total_records': len(df),
            'unique_vehicles': int(df['vehicle_id'].nunique()),
            'duration_seconds': float(df['timestamp'].max()),
            'sampling_rate_hz': 10,
            'sensor_types': ['GPS', 'IMU', 'Compass'],
            'gps_noise_std_meters': 2.0,
            'imu_accel_noise_std': 0.1,
            'imu_gyro_noise_std': 0.01,
        }
        
        import json
        metadata_file = f"{self.output_dir}/metadata_{self.session_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_file
    
    def cleanup(self):
        """Close SUMO"""
        traci.close()
        print("🧹 SUMO closed")

def main():
    print("=" * 60)
    print("🚗 SUMO V2V Dataset Generator with GPS/IMU Sensors")
    print("=" * 60)
    
    collector = SUMODataCollector('config/v2v_100cars.sumocfg')
    
    try:
        collector.start_simulation()
        collector.run_collection(duration_seconds=3600)  # 1 hour
        output_file = collector.save_data()
        
        print("\n🎉 Success! Dataset ready at:")
        print(f"   {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.cleanup()

if __name__ == '__main__':
    main()
