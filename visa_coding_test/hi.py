

dp_array = [0, 0, 1, 2]




def count(value):
    while value >= len(dp_array):
        dp_array.append((len(dp_array) - 1) * (dp_array[-1] + dp_array[-2]))
    return (dp_array[-1] % 1000000007)


def  countWays(arr):
    for value in arr:
        if value == 1:
            print(0)
        elif value == 2:
            print(1)
        elif value == 3:
            print(2)
        else:
            print(count(value))


countWays([1,2,3,4,5,6,7,100])
print(dp_array)


dp_array.append(len(dp_array) - 1) * (dp_array[-1] + (len(dp_array) - 2) * dp_array[-2])
