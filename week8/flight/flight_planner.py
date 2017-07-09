import copy

#Flight Itinerary
#Done by: Chiew Jun Hao and Zhang Hao
class Flight:
    def __init__(self, start_city, start_time, end_city, end_time):
        self.start_city = start_city
        self.start_time = start_time
        self.end_city = end_city
        self.end_time = end_time
    def __str__(self):
        return str((self.start_city, self.start_time))+' -> '+ str((self.end_city, self.end_time))

    def matches(self, city, time):
        return self.start_city == city and self.start_time >= time
    __repr__ = __str__

flightDB = [Flight('Rome', 1, 'Paris', 4),
            Flight('Rome', 3, 'Madrid', 5),
            Flight('Rome', 5, 'Istanbul', 10),
            Flight('Paris', 2, 'London', 4),
            Flight('Paris', 5, 'Oslo', 7),
            Flight('Paris', 5, 'Istanbul', 9),
            Flight('Madrid', 7, 'Rabat', 10),
            Flight('Madrid', 8, 'London', 10),
            Flight('Istanbul', 10, 'Constantinople', 10)]

itinerary=[]

def prt_flights(flights):
    print("################current path###############")
    for i in flights:
        print(i.__str__)
    print("###############################")

def prt_buffer(buffer):
    for i in buffer:
        for j in i:
            print(j.__str__)
        print("###")
#BFS
#each function is a node
def find_itinerary(start_city, start_time, end_city, deadline):
    #store all the flights
    buffer = []

    #marks the node that is barren
    explored = []

    #initialise first split
    for i in flightDB:
        path=[]
        if i.matches(start_city,start_time) and i.end_city not in explored and i.end_time <= deadline:
            path.append(i)
            buffer.append(path)
    # j=buffer[-1][-1]
    # print j.end_city
    #while loop runs algorithm until there is no more path left
    while len(buffer) != 0:
        #takes the last element in the stack, pop it and add start city to explored list
        current_path= copy.deepcopy(buffer[0])
        node = buffer[0][-1]
        print("\n"+node.__str__())
        explored.append(node.start_city)
        buffer.pop(0)
        prt_flights(current_path)

        #exit when path is found
        if node.end_city == end_city:
            print("FFFFFFFFFFFFFFFFFFOOOOOOOOOOOOOOOOOOUND")
            print(len(current_path))
            return prt_result(current_path)

        #expand the nodes and append the new path to the end of the buffer
        for i in flightDB:
            if i.matches(node.end_city,node.end_time) and i.end_city not in explored and i.end_time <= deadline:
                #append because BFS, change if DFS
                newPath = copy.deepcopy(current_path)
                newPath.append(i)
                buffer.append(newPath)
        prt_buffer(buffer)
    return None

def prt_result(path):
    itinerary = ""
    for f in path:
        itinerary += str(f.__str__)
    return itinerary

def find_shortest_itinerary(start_city, end_city):
    for i in range(1,11):
        print("\n#############################"+str(i)+"#########################################")
        result = find_itinerary(start_city,1,end_city,i)
        if result is not None:
            return result
            break


# print find_itinerary('Rome', 1, "London", 10)

############# Part 4 ################
#Yes, it will improve. Say the goal is to go from Rome to Istanbul within 8
# there is a direct flight from Rome to Istanbul that has an end time of 9, it will be identified and algorithm will exit should the deadline >9,
# however, if the deadline is progressively introduced,
# this direct flight node will be pruned. The algorithm will then take on other path and hopefully eventually find a legit path to Istabul
print(find_shortest_itinerary("Rome","Paris"))
