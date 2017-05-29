import sys

primes = [2, 3]
largest = [3]


def find_next():
    largest[0] += 1
    for prime in primes:
        if largest[0] % prime == 0:
            return
        if prime > largest[0]**0.5:
            primes.append(largest[0])
            return
    else:
        primes.append(largest[0])


def prime(n):
    if n % 2 == 0 or n % 3 == 0:
        return "NO"

    if n in primes:
        return "YES"
    else:
        if n > largest[0]:
            while n > largest[0]:
                find_next()
            return prime(n)
        else:
            return "NO"


mode = ''

if mode == 'test':
    f = open('in', 'r')
    fin = f.readlines()
else:
    fin = sys.stdin.readlines()
raw = []
for element in fin:
    raw.append(int(element))

for i in range(1, len(raw)):
    print(prime(raw[i]))
