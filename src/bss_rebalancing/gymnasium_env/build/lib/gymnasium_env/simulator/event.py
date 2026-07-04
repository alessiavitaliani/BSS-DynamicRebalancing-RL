from enum import Enum
from dataclasses import dataclass
from gymnasium_env.simulator.trip import Trip


class EventType(Enum):
    DEPARTURE = "Departure"
    ARRIVAL = "Arrival"


@dataclass(order=True)
class Event:
    time: int
    event_type: EventType
    trip: Trip

    def __post_init__(self):
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an instance of EventType Enum")

    def is_departure(self) -> bool:
        return self.event_type == EventType.DEPARTURE

    def is_arrival(self) -> bool:
        return self.event_type == EventType.ARRIVAL

    def get_trip(self) -> Trip:
        return self.trip
