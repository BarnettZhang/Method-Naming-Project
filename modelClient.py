import zmq
import sys
import os


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    socket.send(str.encode(sys.argv[1]))
    message = socket.recv().decode()
    print(message)
    sys.stdout.flush()

        
        