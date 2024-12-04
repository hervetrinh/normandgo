import pickle
import pandas as pd
from geopy.distance import geodesic

class BikeAvailabilityPredictor:
    def __init__(self, bike_model_path, dock_model_path, station_id_mapping_path):
        """
        Initialize the BikeAvailabilityPredictor with model paths and station ID mapping.

        :param bike_model_path: Path to the bike availability model.
        :param dock_model_path: Path to the dock availability model.
        :param station_id_mapping_path: Path to the station ID mapping file.
        """
        with open(bike_model_path, "rb") as f:
            self.bike_model = pickle.load(f)
        with open(dock_model_path, "rb") as f:
            self.dock_model = pickle.load(f)
        with open(station_id_mapping_path, "rb") as f:
            self.station_id_mapping = pickle.load(f)

    def predict_availability(self, station_id, weekday, hour):
        """
        Predict the availability of bikes and docks for a specific station, weekday, and hour.

        :param station_id: ID of the station.
        :param weekday: Weekday (0=Monday, 6=Sunday).
        :param hour: Hour of the day (0-23).
        :return: Tuple of (bikes_available, docks_available).
        """
        station_id_encoded = self.station_id_mapping.get(str(station_id), None)
        if station_id_encoded is None:
            raise ValueError(f"Station ID {station_id} not found in the mapping.")
        X_new = pd.DataFrame({"weekday": [weekday], "hour": [hour], "station_id_encoded": [station_id_encoded]})
        bikes_available = self.bike_model.predict(X_new)[0]
        docks_available = self.dock_model.predict(X_new)[0]
        return bikes_available, docks_available

    def find_nearest_station_with_bikes(self, start_point, stations, weekday, hour):
        """
        Find the nearest station with predicted bikes available.

        :param start_point: Tuple of (latitude, longitude) for the start point.
        :param stations: List of stations.
        :param weekday: Weekday (0=Monday, 6=Sunday).
        :param hour: Hour of the day (0-23).
        :return: Dictionary representing the nearest station with available bikes.
        """
        sorted_stations = sorted(stations, key=lambda station: geodesic(start_point, (station["latitude"], station["longitude"])).meters)
        for station in sorted_stations:
            bikes_available, _ = self.predict_availability(station["station_id"], weekday, hour)
            if bikes_available > 0:
                return station
        raise ValueError("No stations with available bikes found.")

    def find_nearest_station_with_docks(self, end_point, stations, weekday, hour):
        """
        Find the nearest station with predicted docks available.

        :param end_point: Tuple of (latitude, longitude) for the end point.
        :param stations: List of stations.
        :param weekday: Weekday (0=Monday, 6=Sunday).
        :param hour: Hour of the day (0-23).
        :return: Dictionary representing the nearest station with available docks.
        """
        sorted_stations = sorted(stations, key=lambda station: geodesic(end_point, (station["latitude"], station["longitude"])).meters)
        for station in sorted_stations:
            _, docks_available = self.predict_availability(station["station_id"], weekday, hour)
            if docks_available > 0:
                return station
        raise ValueError("No stations with available docks found.")