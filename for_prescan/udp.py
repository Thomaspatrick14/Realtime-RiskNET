import socket
import numpy as np
import cv2

# Define image dimensions and packet size
image_height = 360
image_width = 640
num_channels = 3
frame_size = image_height * image_width * num_channels  # Total bytes
single_layer = image_height * image_width  # Number of bytes in a single layer

# Create a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('0.0.0.0', 5005))  # Bind to the listening port

print("Waiting for data...")

frame_data = bytearray()
try:
    while True:
        packet, _ = udp_socket.recvfrom(65536)  # Adjust buffer size as needed
        frame_data.extend(packet)
                
        if len(frame_data) == frame_size:
            # Process frame data
            frame = np.stack((
                np.frombuffer(frame_data, dtype=np.uint8)[single_layer*2:].reshape((image_width, image_height)).transpose(),
                np.frombuffer(frame_data, dtype=np.uint8)[single_layer:single_layer*2].reshape((image_width, image_height)).transpose(),
                np.frombuffer(frame_data, dtype=np.uint8)[:single_layer].reshape((image_width, image_height)).transpose()
            ), axis=-1)
            
            cv2.imshow('Received Frame', frame)
        
            # Clear frame_data for next frame
            frame_data.clear()
           
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    udp_socket.close()
    cv2.destroyAllWindows()
