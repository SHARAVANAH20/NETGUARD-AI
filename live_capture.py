
from scapy.all import sniff, IP, TCP, UDP
from datetime import datetime


def process_packet(packet):
    if IP not in packet:
        return

    source_ip = packet[IP].src
    destination_ip = packet[IP].dst
    protocol = packet[IP].proto
    packet_size = len(packet)

    if TCP in packet:
        source_port = packet[TCP].sport
        destination_port = packet[TCP].dport
        protocol_name = "TCP"

    elif UDP in packet:
        source_port = packet[UDP].sport
        destination_port = packet[UDP].dport
        protocol_name = "UDP"

    else:
        source_port = None
        destination_port = None
        protocol_name = str(protocol)

    print(
        f"{datetime.now().strftime('%H:%M:%S')} | "
        f"{protocol_name} | "
        f"{source_ip}:{source_port} -> "
        f"{destination_ip}:{destination_port} | "
        f"Size: {packet_size} bytes"
    )


print("Starting live packet capture...")
print("Press CTRL+C to stop.")

sniff(
    prn=process_packet,
    store=False
)
