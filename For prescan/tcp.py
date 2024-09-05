import socket
import numpy as np
import cv2

# Define image dimensions
image_height = 360
image_width = 480
num_channels = 3
frame_size = image_height * image_width * num_channels  # Total number of bytes in the image
single_layer = image_height * image_width  # Number of bytes in a single layer

# Create a TCP/IP socket
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.bind(('0.0.0.0', 5005))  # Bind to the listening port
tcp_socket.listen(1)

print("Waiting for connection...")
connection, _ = tcp_socket.accept()
print("Connected")

try:
    while True:
        frame_data = bytearray()

        while len(frame_data) < frame_size:
            packet = connection.recv(frame_size - len(frame_data))  # Receive TCP packet
            if not packet:
                break
            frame_data.extend(packet)

        if len(frame_data) == frame_size:
            # Convert the byte data to a NumPy array with dtype uint8 and reshape to (height, width, channels)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            r = frame[:single_layer].reshape((image_width, image_height)).transpose()
            g = frame[single_layer:single_layer*2].reshape((image_width, image_height)).transpose()
            b = frame[single_layer*2:].reshape((image_width, image_height)).transpose()
            frame = np.stack((b, g, r), axis=-1)

            # Display the frame using OpenCV
            cv2.imshow('Received Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    connection.close()
    tcp_socket.close()
    cv2.destroyAllWindows()
