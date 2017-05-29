# Complete the function below.


def move(leftvalue, avg):
    num_of_move = 0
    num_of_right_value = 0
    for i in range(len(avg)):
        if avg[i] != leftvalue:
            num_of_right_value += 1
        if avg[i] == leftvalue:
            num_of_move += num_of_right_value
    return num_of_move




def minMoves(avg):
    print(avg)
    half_len = int(len(avg)/2)
    left_ones = 0
    right_ones = 0
    for i in range(len(avg)):
        if i < half_len:
            if avg[i]==1:
                left_ones += 1

        elif i > half_len:
            if avg[i]==1:
                right_ones += 1


    if left_ones > right_ones:
        #put one on left
        return move(1,avg)
    else:
        #put zero on left
        return move(0,avg)


avg = [1, 1, 1, 1, 0,0,0,0]
print(minMoves(avg))
