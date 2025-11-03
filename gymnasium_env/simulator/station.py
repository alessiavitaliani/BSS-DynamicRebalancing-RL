from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymnasium_env.simulator.bike import Bike
    from gymnasium_env.simulator.cell import Cell

class Station:
    def __init__(self, station_id: int, lat: float, lon: float, name: str = None, capacity: int = 1000,
                 bikes: {"Bike"} = None, request_rate: float = 0.0, arrival_rate: float = 0.0, cell: "Cell" = None):
        """
        Initialize a Station object.

        Parameters:
        station_id (int): Unique identifier for the station.
        name (str): Name of the station.
        lat (float): Latitude of the station's location.
        lon (float): Longitude of the station's location.
        capacity (int): Maximum capacity of bikes the station can hold. Default is 1000.
        bikes (list): List of Bike objects at the station. Default is an empty list.
        request_rate (float): Rate of bike requests at the station. Default is 0.0.
        arrival_rate (float): Rate of bike arrivals at the station. Default is 0.0.
        cell ("Cell"): The parent cell of this station. Default is None.
        """
        self.station_id = station_id
        self.name = name if name is not None else f"Station {station_id}"
        self.lat = lat
        self.lon = lon
        self.capacity = capacity
        self.bikes = bikes if bikes is not None else {}
        self.request_rate = request_rate
        self.arrival_rate = arrival_rate
        self.cell = cell
        self.number_of_bikes = len(self.bikes)

    def __str__(self):
        """
        Return a string representation of the Station object.

        Returns:
        str: A string describing the station with its ID, name, latitude, and longitude.
        """
        return f"Station {self.station_id} (Locked Bikes: {len(self.bikes)})" # Position ({self.lat}, {self.lon}),

    def set_bikes(self, bikes: {"Bike"}):
        """
        Set the list of bikes at the station and upates the respective flags.

        Parameters:
        bikes (list): List of Bike objects at the station.
        """
        if len(bikes) < self.capacity:
            for bike_id, bike in bikes.items():
                bike.set_availability(True)
                bike.set_station(self)
                self.bikes[bike_id] = bike
                self.cell.set_total_bikes(self.cell.get_total_bikes() + 1)
        else:
            raise ValueError("The number of bikes exceeds the station's capacity.")

    def set_request_rate(self, request_rate: float):
        self.request_rate = request_rate

    def set_arrival_rate(self, arrival_rate: float):
        self.arrival_rate = arrival_rate

    def set_capacity(self, capacity: int):
        self.capacity = capacity

    def set_cell(self, cell: "Cell"):
        """
        Set the cell of the station.

        Parameters:
        cell (Cell): The cell object representing the station's location.
        """
        self.cell = cell

    def unlock_bike(self, bike_id: int = None) -> "Bike":
        """
        Unlock a bike from the station and upates the respective flags.

        Returns:
        bike (Bike): The bike to be unlocked from the station.
        """
        if len(self.bikes) > 0:
            if bike_id is None:
                bike_id = max(self.bikes, key=lambda k: self.bikes[k].get_battery())
            bike = self.bikes.pop(bike_id)
            bike.set_availability(False)
            bike.set_station(None)
            self.cell.set_total_bikes(self.cell.get_total_bikes() - 1)
            return bike
        else:
            raise ValueError("Station is empty. Cannot unlock bike.")

    def lock_bike(self, bike: "Bike"):
        """
        Lock a bike at the station and upates the respective flags.

        Parameters:
        bike (Bike): The bike to be locked at the station.
        """
        if len(self.bikes) < self.capacity:
            bike.set_availability(True)
            bike.set_station(self)
            self.bikes[bike.get_bike_id()] = bike
            self.cell.set_total_bikes(self.cell.get_total_bikes() + 1)
        else:
            raise ValueError("Station is full. Cannot lock bike. Dimension: " + str(len(self.bikes)) + "/" + str(self.capacity))

    def get_station_id(self) -> int:
        return self.station_id

    def get_name(self) -> str:
        return self.name

    def get_coordinates(self) -> (float, float):
        return self.lat, self.lon

    def get_bikes(self) -> {"Bike"}:
        return self.bikes

    def get_request_rate(self) -> float:
        return self.request_rate

    def get_arrival_rate(self) -> float:
        return self.arrival_rate

    def get_cell(self) -> "Cell":
        return self.cell

    def get_capacity(self) -> int:
        return self.capacity

    def get_number_of_bikes(self) -> int:
        return len(self.bikes)