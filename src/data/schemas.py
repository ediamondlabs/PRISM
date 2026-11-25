from typing import List

from pydantic import BaseModel, Field, field_validator


class VehicleRecord(BaseModel):
    """Schema for a single vehicle sensor record."""

    frame: int = Field(gt=0, description="Frame number")
    timestamp: float = Field(ge=0.0, description="Timestamp in seconds")
    vehicle_id: int = Field(ge=0, description="Unique vehicle identifier")
    latitude: float = Field(description="GPS latitude")
    longitude: float = Field(description="GPS longitude")
    altitude: float = Field(description="GPS altitude in meters")
    x: float = Field(description="Cartesian X coordinate")
    y: float = Field(description="Cartesian Y coordinate")
    z: float = Field(description="Cartesian Z coordinate")
    roll: float = Field(description="Roll angle")
    pitch: float = Field(description="Pitch angle")
    yaw: float = Field(description="Yaw angle")
    velocity_x: float = Field(description="Velocity in X direction (m/s)")
    velocity_y: float = Field(description="Velocity in Y direction (m/s)")
    velocity_z: float = Field(description="Velocity in Z direction (m/s)")
    speed: float = Field(ge=0.0, description="Speed magnitude (m/s)")
    accel_x: float = Field(description="Acceleration in X direction (m/s²)")
    accel_y: float = Field(description="Acceleration in Y direction (m/s²)")
    accel_z: float = Field(description="Acceleration in Z direction (m/s²)")
    gyro_x: float = Field(description="Gyroscope X reading (rad/s)")
    gyro_y: float = Field(description="Gyroscope Y reading (rad/s)")
    gyro_z: float = Field(description="Gyroscope Z reading (rad/s)")
    compass: float = Field(description="Compass heading")


class TrajectoryFeatures(BaseModel):
    """Schema for trajectory window features (10 features)."""

    velocity_x: List[float] = Field(description="X velocity sequence")
    velocity_y: List[float] = Field(description="Y velocity sequence")
    speed: List[float] = Field(description="Speed sequence")
    accel_x: List[float] = Field(description="X acceleration sequence")
    accel_y: List[float] = Field(description="Y acceleration sequence")
    accel_z: List[float] = Field(description="Z acceleration sequence")
    gyro_x: List[float] = Field(description="X gyroscope sequence")
    gyro_y: List[float] = Field(description="Y gyroscope sequence")
    gyro_z: List[float] = Field(description="Z gyroscope sequence")
    yaw: List[float] = Field(description="Yaw angle sequence")

    @field_validator("*")
    @classmethod
    def check_sequence_length(cls, v: List[float]) -> List[float]:
        """Validate that all sequences have consistent length."""
        if not v:
            raise ValueError("Feature sequence cannot be empty")
        return v


class DatasetMetadata(BaseModel):
    """Schema for dataset metadata."""

    session_id: str
    total_records: int = Field(gt=0)
    unique_vehicles: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0)
    sampling_rate_hz: int = Field(gt=0)
    sensor_types: List[str]
    gps_noise_std_meters: float = Field(ge=0.0)
    imu_accel_noise_std: float = Field(ge=0.0)
    imu_gyro_noise_std: float = Field(ge=0.0)


class VehicleStats(BaseModel):
    """Statistics for a single vehicle."""

    vehicle_id: int = Field(ge=0)
    num_records: int = Field(gt=0)
    presence_duration: float = Field(gt=0.0, description="Duration in seconds")
    avg_speed: float = Field(ge=0.0, description="Average speed (m/s)")
    max_speed: float = Field(ge=0.0, description="Maximum speed (m/s)")
    trajectory_length: float = Field(ge=0.0, description="Total distance traveled (m)")

    @field_validator("presence_duration")
    @classmethod
    def validate_presence(cls, v: float) -> float:
        """Ensure presence duration is positive."""
        if v <= 0:
            raise ValueError("Presence duration must be positive")
        return v
