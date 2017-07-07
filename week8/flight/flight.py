class Flight:
    def __init__(self, start_city, start_time, end_city, end_time):
        self.start_city = start_city
        self.start_time = start_time
        self.end_city = end_city
        self.end_time = end_time

    def __str__(self):
        return str((self.start_city, self.start_time)) + ' -> ' + str((self.end_city, self.end_time))

    __repr__ = __str__

    # Write a method matches contained within the Flight class, that
    # takes a pair (city,time) as an argument and returns a boolean based
    # on whether or not the city and time specified "match" those of the
    # flight, in the sense that the flight leaves from the same city, at a time
    # later than time.
    def match(self, city, time):
        return city == self.start_city and time >= self.start_time


flightDB = [Flight('Rome', 1, 'Paris', 4),
            Flight('Rome', 3, 'Madrid', 5),
            Flight('Rome', 5, 'Istanbul', 10),
            Flight('Paris', 2, 'London', 4),
            Flight('Paris', 5, 'Oslo', 7),
            Flight('Paris', 5, 'Istanbul', 9),
            Flight('Madrid', 7, 'Rabat', 10),
            Flight('Madrid', 8, 'London', 10),
            Flight('Istanbul', 10, 'Constantinople', 10)]


class Itinerary:
    itinerary = []

    def add(self, flight):
        self.itinerary.append(flight)


class State:
    time = None
    city = None
    itinerary = None

    def __init__(self, time, city, itinerary=None):
        pass

    def get_state(self):
        return self

    def is_same(self, other):
        if len(self.itinerary.get_itinerary()) != len(other.get_state().itinerary.itinerary):
            return False
        else:
            for i in range(len(self.itinerary.itinerary_list)):
                if self.itinerary.itinerary_list[i] != other.get_state().itinerary.itinerary[i]:
                    return False
        if self.time != other.get_state.time:
            return False

        if self.city != other.get_state.city:
            return False

        return True


# Define a procedure find_itinerary that returns a plan, in the
# form of a sequence of (city, time) pairs, that represents a legal sequence
# of flights (found in FlightDB) from start_city to end_city
# before a specified deadline.
def find_itinerary(start_city, start_time, end_city, deadline):
    current_city = start_city
    current_time = start_time
    frontier = []

    pass
