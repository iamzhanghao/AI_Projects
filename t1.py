def g(z):
    if abs(z)<1:
        return z
    elif z>=1:
        return 1
    else:
        return -1


def z(w0,w1,x1,w2,x2):
    return -w0 + w1*x1 +w2 * x2


solutions = [[0,0,1],[1,1,1],[0,1,-1],[1,0,-1]]
progress = 0

range_max = 3

for w11 in range(-range_max,range_max):
    for w12 in range(-range_max,range_max):
        for w21 in range(-range_max,range_max):
            for w22 in range(-range_max,range_max):
                for w1 in range(-range_max,range_max):
                    for w2 in range(-range_max,range_max):

                        correct = True
                        for solution in solutions:
                            x1 = solution[0]
                            x2 = solution[1]
                            y = solution[2]
                            z1 = g(z(-1, w11, x1, w21, x2))
                            z2 = g(z(-1, w12, x1, w22, x2))
                            yout = g(z(-1, w1, z1, w2, z2))
                            if yout != y:
                                correct = False
                        if correct is True:
                            print(w1,w2,w11,w12,w21,w22)


