import datetime
from geopy.distance import geodesic
from route_finder import RouteFinder
from bike_availability_predictor import BikeAvailabilityPredictor

class RouteFinderApp:
    def __init__(self, config):
        self.route_finder = RouteFinder(config['osrm']['url'], config['gtfs']['path'], config['station_info']['url'], config['station_status']['url'])
        self.bike_predictor = BikeAvailabilityPredictor(
            config['data']['models']['bike_model_path'],
            config['data']['models']['dock_model_path'],
            config['data']['models']['station_id_mapping_path']
        )

    def find_routes(self, data):
        start_point = (data['start_lat'], data['start_lng'])
        end_point = (data['end_lat'], data['end_lng'])
        mode = data.get('mode', 'bike')

        if mode == "bike":
            stations = self.route_finder.get_vls_stations()
            start_station = self.route_finder.find_nearest_station(start_point, stations, available_bikes=True)
            end_station = self.route_finder.find_nearest_station(end_point, stations, available_bikes=False)
            weekday = datetime.datetime.now().weekday()
            hour = datetime.datetime.now().hour
            bikes_available, _ = self.bike_predictor.predict_availability(start_station["station_id"], weekday, hour)
            _, docks_available = self.bike_predictor.predict_availability(end_station["station_id"], weekday, hour)
            if bikes_available < 1:
                start_station = self.bike_predictor.find_nearest_station_with_bikes(start_point, stations, weekday, hour)
            if docks_available < 1:
                end_station = self.bike_predictor.find_nearest_station_with_docks(end_point, stations, weekday, hour)
            walk_to_start = self.route_finder.get_route(start_point, (start_station["latitude"], start_station["longitude"]))
            bike_route = self.route_finder.get_route((start_station["latitude"], start_station["longitude"]),
                                                     (end_station["latitude"], end_station["longitude"]))
            walk_to_end = self.route_finder.get_route((end_station["latitude"], end_station["longitude"]), end_point)
            walk_to_start_distance_km = sum(
                geodesic(walk_to_start[i], walk_to_start[i + 1]).km for i in range(len(walk_to_start) - 1)
            )
            bike_distance_km = sum(
                geodesic(bike_route[i], bike_route[i + 1]).km for i in range(len(bike_route) - 1)
            )
            walk_to_end_distance_km = sum(
                geodesic(walk_to_end[i], walk_to_end[i + 1]).km for i in range(len(walk_to_end) - 1)
            )
            coins_earned = self.route_finder.route_coin(walk_to_start_distance_km + bike_distance_km + walk_to_end_distance_km, mode="bike")
            return {
                "start_station": {
                    "name": start_station["name"],
                    "latitude": start_station["latitude"],
                    "longitude": start_station["longitude"],
                    "available_bikes": bikes_available,
                    "available_stands": start_station["available_stands"]
                },
                "end_station": {
                    "name": end_station["name"],
                    "latitude": end_station["latitude"],
                    "longitude": end_station["longitude"],
                    "available_bikes": end_station["available_bikes"],
                    "available_stands": docks_available
                },
                "routes": {
                    "walk_to_start": walk_to_start,
                    "bike_route": bike_route,
                    "walk_to_end": walk_to_end
                },
                "co2_emissions": 0.0,
                "transport_mode": "velo",
                "coins_earned": coins_earned
            }
        elif mode == "bus":
            bus_route = self.route_finder.get_bus_route(start_point, end_point)
            bus_route = self.route_finder.clean_dict(bus_route)
            if "error" in bus_route:
                return {"error": "Nous n'avons pas trouvé de ligne de transport en commun qui lie les deux adresses."}, 404
            walk_to_start_distance_km = sum(
                geodesic(bus_route["routes"]["walk_to_start"][i], bus_route["routes"]["walk_to_start"][i + 1]).km
                for i in range(len(bus_route["routes"]["walk_to_start"]) - 1)
            )
            bus_distance_km = sum(
                geodesic(bus_route["routes"]["public_transport_route"][i],
                         bus_route["routes"]["public_transport_route"][i + 1]).km
                for i in range(len(bus_route["routes"]["public_transport_route"]) - 1)
            )
            walk_to_end_distance_km = sum(
                geodesic(bus_route["routes"]["walk_to_end"][i], bus_route["routes"]["walk_to_end"][i + 1]).km
                for i in range(len(bus_route["routes"]["walk_to_end"]) - 1)
            )
            coins_earned = self.route_finder.route_coin(walk_to_start_distance_km + bus_distance_km + walk_to_end_distance_km, mode="bus")
            bus_route["coins_earned"] = coins_earned
            return bus_route
        else:
            return {"error": "Invalid transportation mode"}, 400