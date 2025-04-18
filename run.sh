#!/bin/bash

SCENARIO_NAME="DynamicObjectCrossing_4"
# MAIN_PY_ARGS="--no-save" 
MAIN_PY_ARGS="" 

tmux new-session -d -s carla_session "python scenario_runner/scenario_runner.py --scenario ${SCENARIO_NAME} --reloadWorld"
sleep 3
tmux split-window -t carla_session "python main.py ${MAIN_PY_ARGS} --scenario-name ${SCENARIO_NAME}"
tmux attach -t carla_session

# Currently the following scenarios are supported:

# ManeuverOppositeDirection_1
# ManeuverOppositeDirection_2
# ManeuverOppositeDirection_3
# ManeuverOppositeDirection_4
# SignalizedJunctionLeftTurn_1
# SignalizedJunctionLeftTurn_2
# SignalizedJunctionLeftTurn_3
# SignalizedJunctionLeftTurn_4
# SignalizedJunctionLeftTurn_5
# SignalizedJunctionLeftTurn_6
# ChangeLane_1
# ChangeLane_2
# FollowLeadingVehicle_1
# FollowLeadingVehicleWithObstacle_1
# FollowLeadingVehicle_2
# FollowLeadingVehicleWithObstacle_2
# FollowLeadingVehicle_3
# FollowLeadingVehicleWithObstacle_3
# FollowLeadingVehicle_4
# FollowLeadingVehicleWithObstacle_4
# FollowLeadingVehicle_5
# FollowLeadingVehicleWithObstacle_5
# FollowLeadingVehicle_6
# FollowLeadingVehicleWithObstacle_6
# FollowLeadingVehicle_7
# FollowLeadingVehicleWithObstacle_7
# FollowLeadingVehicle_8
# FollowLeadingVehicleWithObstacle_8
# FollowLeadingVehicle_9
# FollowLeadingVehicleWithObstacle_9
# FollowLeadingVehicle_10
# FollowLeadingVehicleWithObstacle_10
# FollowLeadingVehicle_11
# FollowLeadingVehicleWithObstacle_11
# StationaryObjectCrossing_1
# DynamicObjectCrossing_1
# StationaryObjectCrossing_2
# DynamicObjectCrossing_2
# StationaryObjectCrossing_3
# DynamicObjectCrossing_3
# StationaryObjectCrossing_4
# DynamicObjectCrossing_4
# StationaryObjectCrossing_5
# DynamicObjectCrossing_5
# StationaryObjectCrossing_6
# DynamicObjectCrossing_6
# StationaryObjectCrossing_7
# DynamicObjectCrossing_7
# StationaryObjectCrossing_8
# DynamicObjectCrossing_8
# DynamicObjectCrossing_9
# ConstructionSetupCrossing
# CutInFrom_left_Lane
# CutInFrom_right_Lane
# OtherLeadingVehicle_1
# OtherLeadingVehicle_2
# OtherLeadingVehicle_3
# OtherLeadingVehicle_4
# OtherLeadingVehicle_5
# OtherLeadingVehicle_6
# OtherLeadingVehicle_7
# OtherLeadingVehicle_8
# OtherLeadingVehicle_9
# OtherLeadingVehicle_10
# OppositeVehicleRunningRedLight_1
# OppositeVehicleRunningRedLight_2
# OppositeVehicleRunningRedLight_3
# OppositeVehicleRunningRedLight_4
# OppositeVehicleRunningRedLight_5
# VehicleTurningRight_1
# VehicleTurningLeft_1
# VehicleTurningRight_2
# VehicleTurningLeft_2
# VehicleTurningRight_3
# VehicleTurningLeft_3
# VehicleTurningRight_4
# VehicleTurningLeft_4
# VehicleTurningRight_5
# VehicleTurningLeft_5
# VehicleTurningRight_6
# VehicleTurningLeft_6
# VehicleTurningRight_7
# VehicleTurningLeft_7
# VehicleTurningRight_8
# VehicleTurningLeft_8
# SignalizedJunctionRightTurn_1
# SignalizedJunctionRightTurn_2
# SignalizedJunctionRightTurn_3
# SignalizedJunctionRightTurn_4
# SignalizedJunctionRightTurn_5
# SignalizedJunctionRightTurn_6
# SignalizedJunctionRightTurn_7
# FreeRide_1
# FreeRide_2
# FreeRide_3
# FreeRide_4
# MultiEgo_1
# MultiEgo_2
# ControlLoss_1
# ControlLoss_2
# ControlLoss_3
# ControlLoss_4
# ControlLoss_5
# ControlLoss_6
# ControlLoss_7
# ControlLoss_8
# ControlLoss_9
# ControlLoss_10
# ControlLoss_11
# ControlLoss_12
# ControlLoss_13
# ControlLoss_14
# ControlLoss_15
# NoSignalJunctionCrossing
# CARLA:PedestrianCrossing (OpenSCENARIO)
# CARLA:ChangingWeatherExample (OpenSCENARIO)
# CARLA:FollowLeadingVehicle (OpenSCENARIO)
# CARLA:ControllerExample (OpenSCENARIO)
# CARLA:LaneChangeSimple (OpenSCENARIO)
# CARLA:Slalom (OpenSCENARIO)
# CARLA:CyclistCrossing (OpenSCENARIO)
# CARLA:SynchronizeIntersectionEntry (OpenSCENARIO)
# CARLA:PedestrianCrossing (OpenSCENARIO)
