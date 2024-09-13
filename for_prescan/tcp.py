import socket
import numpy as np
import cv2

# Define image dimensions
image_height = 360
image_width = 640
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
frame_data = bytearray()
i=0
try:
    while True:

        while len(frame_data) < frame_size:
            packet = connection.recv(frame_size - len(frame_data))  # Receive TCP packet
            if not packet:
                break
            frame_data.extend(packet)

        if len(frame_data) == frame_size:
            # Convert the byte data to a NumPy array with dtype uint8 and reshape to (height, width, channels)
            frame = np.stack((np.frombuffer(frame_data, dtype=np.uint8)[single_layer*2:].reshape((image_width, image_height)).transpose(),
                              np.frombuffer(frame_data, dtype=np.uint8)[single_layer:single_layer*2].reshape((image_width, image_height)).transpose(),
                              np.frombuffer(frame_data, dtype=np.uint8)[:single_layer].reshape((image_width, image_height)).transpose()), axis=-1)

            # Clear frame_data for next frame
            frame_data.clear()
            cv2.imshow('Received Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    connection.close()
    tcp_socket.close()
    cv2.destroyAllWindows()
