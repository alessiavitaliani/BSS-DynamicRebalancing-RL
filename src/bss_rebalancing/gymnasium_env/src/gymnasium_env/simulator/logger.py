import math
import logging
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.trip import Trip


class Logger:
    def __init__(self, log_file: str, is_logging: bool = False):
        logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')
        self.logger = logging.getLogger('env_logger')
        self.is_logging = is_logging

    def set_logging(self, is_logging: bool):
        self.is_logging = is_logging

    def new_log_line(self, timeslot=None):
        if self.is_logging:
            log = "--------------------------------------------------------"
            if timeslot is not None:
                log += f" Timeslot = {timeslot}"
            self.logger.info(log)

    def log_starting_action(self, action: str, t: int, cell_id: int, invalid: bool):
        if self.is_logging:
            invalid = 'INVALID' if invalid else ''
            self.logger.info(
                f'ACTION: {action} on cell {cell_id} {invalid} --> Time to complete: {t}s - Steps needed: {int(math.ceil(t / 30))}')

    def log_ending_action(self, invalid: bool, time: str):
        if self.is_logging:
            if invalid:
                self.logger.info(f'### ACTION IS INVALID ### - Time: {time}')
            else:
                self.logger.info(f'Action completed successfully - Time: {time}')

    def log_state(self, step: int, time: str):
        if self.is_logging:
            self.logger.info(f'State S_{step} - Time: {time}')

    def log_truck(self, truck: Truck, depot_bikes: int):
        if self.is_logging:
            self.logger.info(
                f"TRUCK in CELL: {truck.cell.get_id()} (center_node = {truck.cell.get_center_node()}, cell_bikes = {truck.cell.get_total_bikes()}, critic_score = {truck.cell.get_critic_score()})"
                f" - POSITION: {truck.position} - LOAD: {truck.current_load}-{depot_bikes}")

    def log_no_available_bikes(self, start_station: Station, end_station: Station):
        if self.is_logging:
            self.logger.warning(
                f"TRIP FAILED: No bikes from station {start_station.get_station_id()} (cell: {start_station.get_cell().get_id()})"
                f" to station {end_station.get_station_id()} (cell: {end_station.get_cell().get_id()})")

    def log_trip(self, trip: Trip):
        if self.is_logging:
            self.logger.info("Trip: %s", trip)

    def log_terminated(self, time: str):
        if self.is_logging:
            self.logger.info('#################################################'
                             f'###### Slot TERMINATED - Time: {time} '
                             '#################################################')

    def log_done(self, time: str):
        if self.is_logging:
            self.logger.info('#################################################'
                             f'###### THE EPISODE IS DONE - Time: {time} '
                             '#################################################')

    def log(self, message: str):
        if self.is_logging:
            self.logger.info(f"{message}")

    def warning(self, message: str):
        if self.is_logging:
            self.logger.warning(f"{message}")