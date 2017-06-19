import socket

host = "10.12.44.104"
port = 5001

mySocket = socket.socket()
mySocket.bind((host, port))

print("Listening for connection...")

mySocket.listen(1)
conn, addr = mySocket.accept()
print("Connection from: " + str(addr))
while True:
    data = conn.recv(1024).decode()
    if not data:
        break
    print("from connected  user: " + str(data))

conn.close()