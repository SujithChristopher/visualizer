import socket
import time
import struct
# Create a UDP socket
local_ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

local_ip.connect(("8.8.8.8", 80))

# Get the local IPv4 address from the socket
local_ip_address = local_ip.getsockname()[0]

# Define the broadcast ip and port
broadcast_ip = ''.join([ip + "." for ip in local_ip_address.split(".")[:-1]]) + "255" # getting broadcast ip address
broadcast_port = 23000

time.sleep(1)
ttl = 1
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, ttl)



message = "A"

udp_socket.sendto(message.encode(), (broadcast_ip, broadcast_port))
print(f"Multicast Sent: {message}")
try:
    while True:
        # Send the multicast message
        return_message = udp_socket.recvfrom(23000)
        print(f"Multicast Received: {return_message[0]}")
        print(f"Multicast Received: {struct.unpack('4l', return_message[0][4:20])}")
        
except KeyboardInterrupt:
    print("Multicast sender stopped by user.")
finally:
    # Close the socket when done
    udp_socket.close()
    
    
    
