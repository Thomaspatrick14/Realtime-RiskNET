import socket
import numpy as np
import cv2

# Define image dimensions and packet size
image_height = 480
image_width = 640
num_channels = 3
frame_size = image_height * image_width * num_channels  # Total bytes
udp_packet_size = 65400  # Adjust based on your Simulink configuration

# Create a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('0.0.0.0', 5005))  # Bind to the listening port

print("Waiting for data...")

try:
    while True:
        frame_data = bytearray()

        while len(frame_data) < frame_size:
            packet, _ = udp_socket.recvfrom(udp_packet_size)  # Receive data
            frame_data.extend(packet)

        if len(frame_data) == frame_size:
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            r = frame[:307200].reshape((image_width, image_height)).transpose()
            g = frame[307200:614400].reshape((image_width, image_height)).transpose()
            b = frame[614400:].reshape((image_width, image_height)).transpose()
            frame = np.stack((b, g, r), axis=-1)
            cv2.imshow('Received Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    udp_socket.close()
    cv2.destroyAllWindows()
