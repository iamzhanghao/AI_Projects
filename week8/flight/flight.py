import copy


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


class State:
    time = None
    city = None
    itinerary = []

    def __init__(self, time, city, itinerary=[]):
        self.time = time
        self.city = city
        self.itinerary = itinerary

    def is_same(self, other):
        return self.time == other.time and self.city == other.time

    def get_Itinerary(self):
        return copy.deepcopy(self.itinerary)


def bfs(frontier, explored_state, end_city, deadline):
    if len(frontier) == 0:
        return None

    if frontier[0].city == end_city:
        return frontier[0].itinerary
    else:
        current_state = frontier.pop(0)
        for flight in flightDB:
            if flight.start_city == current_state.city and flight.start_time >= current_state.time and flight.end_time <= deadline \
                    and (flight.end_city, flight.end_time) not in explored_state:
                # print("add",flight)
                new_state = State(flight.end_time, flight.end_city, current_state.get_Itinerary())
                new_state.itinerary.append(flight)
                frontier.append(new_state)
                explored_state.add((flight.end_city, flight.end_time))

    return bfs(frontier, explored_state, end_city, deadline)


def find_itinerary(start_city, start_time, end_city, deadline):
    current_city = start_city
    current_time = start_time
    frontier = [State(current_time, current_city)]
    explored_state = set()
    explored_state.add((current_city, current_time))

    return bfs(frontier, explored_state, end_city, deadline)


print("\nUsing find_itinerary")

itinerary = find_itinerary('Rome', 1, 'Constantinople', 10)

if itinerary is not None:
    for flight in itinerary:
        print(flight)
else:
    print("No Solution")


def find_shortest_itinerary(start_city, end_city):
    for i in range(1, 12):
        itinerary = find_itinerary(start_city, 1, end_city, i)
        if itinerary is not None:
            return itinerary

    return None


### METHOD 1 ###
# This guarantee to return the soonest flight
print("\nUsing find_shortest_itinerary")
itinerary = find_shortest_itinerary('Rome', 'London')

if itinerary is not None:
    for flight in itinerary:
        print(flight)
else:
    print("No Solution")


### METHOD 2 ###
# Reverse look up
def reverse_shortest_itinerary(start_city,end_city):
    itineraries = []
    for i in range(11, 0, -1):
        itinerary = find_itinerary(start_city, 1, end_city, i)
        if itinerary is not None:
            itineraries.append(itinerary)

    if len(itineraries)!= 0:
        return itineraries[0]
    else:
        return None

print("\nUsing reverse_shortest_itinerary")
itinerary = reverse_shortest_itinerary('Rome', 'London')

if itinerary is not None:
    for flight in itinerary:
        print(flight)
else:
    print("No Solution")







