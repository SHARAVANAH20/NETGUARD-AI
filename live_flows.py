
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
from datetime import datetime
import time
import threading


flows = defaultdict(lambda: {
    "start_time": time.time(),
    "last_time": time.time(),
    "packet_count": 0,
    "total_bytes": 0,
    "forward_packets": 0,
    "backward_packets": 0,
    "syn_count": 0,
    "ack_count": 0,
    "rst_count": 0,
    "fin_count": 0
})

flow_lock = threading.Lock()


def process_packet(packet):
    if IP not in packet:
        return

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    if TCP in packet:
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        protocol = "TCP"

    elif UDP in packet:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
        protocol = "UDP"

    else:
        return

    forward_key = (src_ip, src_port, dst_ip, dst_port, protocol)
    reverse_key = (dst_ip, dst_port, src_ip, src_port, protocol)

    with flow_lock:
        if forward_key in flows:
            flow_key = forward_key
            direction = "forward"
        elif reverse_key in flows:
            flow_key = reverse_key
            direction = "backward"
        else:
            flow_key = forward_key
            direction = "forward"

        flow = flows[flow_key]
        flow["last_time"] = time.time()
        flow["packet_count"] += 1
        flow["total_bytes"] += len(packet)

        if direction == "forward":
            flow["forward_packets"] += 1
        else:
            flow["backward_packets"] += 1

        if TCP in packet:
            flags = packet[TCP].flags

            if flags & 0x02:
                flow["syn_count"] += 1
            if flags & 0x10:
                flow["ack_count"] += 1
            if flags & 0x04:
                flow["rst_count"] += 1
            if flags & 0x01:
                flow["fin_count"] += 1


def print_flows():
    while True:
        time.sleep(10)

        with flow_lock:
            print("\n========== LIVE NETWORK FLOWS ==========")

            for key, flow in list(flows.items()):
                duration = max(
                    flow["last_time"] - flow["start_time"],
                    0.001
                )

                print(
                    f"{key[0]}:{key[1]} -> {key[2]}:{key[3]} "
                    f"| {key[4]} "
                    f"| Packets: {flow['packet_count']} "
                    f"| Bytes: {flow['total_bytes']} "
                    f"| Duration: {duration:.2f}s "
                    f"| SYN: {flow['syn_count']} "
                    f"| ACK: {flow['ack_count']} "
                    f"| RST: {flow['rst_count']}"
                )


print("Starting live flow capture...")
print("Flow summaries are printed every 10 seconds.")
print("Press CTRL+C to stop.")

threading.Thread(
    target=print_flows,
    daemon=True
).start()

sniff(
    prn=process_packet,
    store=False
)
