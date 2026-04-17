from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymnasium_env.simulator.station import Station


class Bike:
    # Class variable to track the next available bike_id
    _next_bike_id = 0

    def __init__(self, station: 'Station' = None, max_battery: float = 50.0, bike_id: int = None):
        """
        Initialize a Bike object.

        Parameters:
        station (Station): The station where the bike is located.
        max_battery (int): Maximum battery capacity of the bike. Default is 100 (in km).
        bike_id (int): Unique identifier for the bike.
        """

        # Auto-increment bike_id if not provided
        if bike_id is None:
            self.bike_id = Bike._next_bike_id
            Bike._next_bike_id += 1
        else:
            self.bike_id = bike_id
            # Update the counter if manually assigned ID is higher
            if bike_id >= Bike._next_bike_id:
                Bike._next_bike_id = bike_id + 1

        self.station = station
        self.max_battery = max_battery
        self.battery = max_battery
        self.available = False

    def __str__(self):
        """
        Return a string representation of the Bike object.

        Returns:
        str: A string describing the bike with its ID and current station.
        """
        return f"Bike {self.bike_id} at {self.station} - Battery: {self.battery} km - Available: {self.available}"

    @classmethod
    def reset_bike_id_counter(cls, start_id: int = 0):
        """Reset the bike_id counter to a specific value."""
        cls._next_bike_id = start_id

    @classmethod
    def get_next_bike_id(cls) -> int:
        """Get the next bike_id that will be assigned."""
        return cls._next_bike_id

    def set_availability(self, available: bool):
        """
        Set the availability status of the bike.

        Parameters:
        available (bool): True if the bike is available, False otherwise.
        """
        self.available = available

    def set_station(self, station: 'Station'):
        """
        Set the station where the bike is located.

        Parameters:
        stn (Station): The station where the bike is located.
        """
        self.station = station

    def set_battery(self, battery: float):
        """
        Set the battery level of the bike.

        Parameters:
        battery (int): The battery level of the bike.
        """
        self.battery = battery

    def get_station(self) -> 'Station':
        """
        Get the station where the bike is located.

        Returns:
        Station: The station where the bike is located.
        """
        return self.station

    def get_battery(self) -> float:
        """
        Get the battery level of the bike.

        Returns:
        int: The battery level of the bike.
        """
        return self.battery

    def get_bike_id(self) -> int:
        """
        Get the ID of the bike.

        Returns:
        int: The ID of the bike.
        """
        return self.bike_id

    def get_availability(self) -> bool:
        """
        Get the availability status of the bike.

        Returns:
        bool: True if the bike is available, False otherwise.
        """
        return self.available

    def get_max_battery(self) -> float:
        """
        Get the maximum battery capacity of the bike.

        Returns:
        float: The maximum battery capacity of the bike.
        """
        return self.max_battery

    def reset(self, station: 'Station' = None, battery: float = None, available: bool = False):
        # Reset the bike to its initial state
        self.battery = self.max_battery if battery is None else battery
        self.available = available
        self.station = station