
from scapy.all import sniff, IP, TCP, UDP
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
import threading
import time
import numpy as np


OUTPUT_DIR = Path("data/live")
OUTPUT_FILE = OUTPUT_DIR / "live_flows_v2.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_INTERVAL = 10
FLOW_IDLE_TIMEOUT = 60

flows = {}
flow_lock = threading.Lock()


def create_flow(src_ip, dst_ip, src_port, dst_port, protocol, current_time):

    return {
        "Source IP": src_ip,
        "Destination IP": dst_ip,
        "Source Port": src_port,
        "Destination Port": dst_port,
        "Protocol": protocol,
        "Start Time": current_time,
        "Last Seen": current_time,
        "Forward Packet Lengths": [],
        "Backward Packet Lengths": [],
        "Forward Timestamps": [],
        "Backward Timestamps": [],
        "Total Packets": 0,
        "Total Bytes": 0,
        "SYN Flag Count": 0,
        "ACK Flag Count": 0,
        "FIN Flag Count": 0,
        "RST Flag Count": 0,
        "PSH Flag Count": 0,
        "URG Flag Count": 0
    }


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

    current_time = time.time()
    packet_size = len(packet[IP])

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

            flows[flow_key] = create_flow(
                src_ip, dst_ip, src_port, dst_port,
                protocol, current_time
            )

        flow = flows[flow_key]

        flow["Last Seen"] = current_time
        flow["Total Packets"] += 1
        flow["Total Bytes"] += packet_size

        if direction == "forward":
            flow["Forward Packet Lengths"].append(packet_size)
            flow["Forward Timestamps"].append(current_time)

        else:
            flow["Backward Packet Lengths"].append(packet_size)
            flow["Backward Timestamps"].append(current_time)

        if TCP in packet:

            flags = packet[TCP].flags

            if flags & 0x02:
                flow["SYN Flag Count"] += 1

            if flags & 0x10:
                flow["ACK Flag Count"] += 1

            if flags & 0x01:
                flow["FIN Flag Count"] += 1

            if flags & 0x04:
                flow["RST Flag Count"] += 1

            if flags & 0x08:
                flow["PSH Flag Count"] += 1

            if flags & 0x20:
                flow["URG Flag Count"] += 1


def calculate_statistics(values):

    if not values:
        return {
            "Max": 0,
            "Min": 0,
            "Mean": 0,
            "Std": 0
        }

    return {
        "Max": max(values),
        "Min": min(values),
        "Mean": float(np.mean(values)),
        "Std": float(np.std(values, ddof=1)) if len(values) > 1 else 0
    }



def calculate_iat(timestamps):

    if len(timestamps) < 2:
        return {
            "Mean": 0,
            "Std": 0,
            "Max": 0,
            "Min": 0
        }
    timestamps = sorted(timestamps)
    intervals = np.diff(timestamps) * 1_000_000

    return {
        "Mean": float(np.mean(intervals)),
        "Std": float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0,
        "Max": float(np.max(intervals)),
        "Min": float(np.min(intervals))
    }


def export_flows():

    while True:

        time.sleep(EXPORT_INTERVAL)
        current_time = time.time()

        with flow_lock:

            rows = []

            for flow in flows.values():

                duration = max(
                    flow["Last Seen"] - flow["Start Time"],
                    0
                )

                fwd = flow["Forward Packet Lengths"]
                bwd = flow["Backward Packet Lengths"]

                fwd_stats = calculate_statistics(fwd)
                bwd_stats = calculate_statistics(bwd)

                fwd_iat = calculate_iat(flow["Forward Timestamps"])
                bwd_iat = calculate_iat(flow["Backward Timestamps"])

                all_lengths = fwd + bwd
                all_stats = calculate_statistics(all_lengths)

                row = {
                    "Source IP": flow["Source IP"],
                    "Destination IP": flow["Destination IP"],
                    "Source Port": flow["Source Port"],
                    "Destination Port": flow["Destination Port"],
                    "Protocol": flow["Protocol"],
                    "Start Time": flow["Start Time"],
                    "Last Seen": flow["Last Seen"],
                    "Flow Duration": duration * 1_000_000,
                    "Total Fwd Packets": len(fwd),
                    "Total Backward Packets": len(bwd),
                    "Total Length of Fwd Packets": sum(fwd),
                    "Total Length of Bwd Packets": sum(bwd),
                    "Total Packets": flow["Total Packets"],
                    "Total Bytes": flow["Total Bytes"],
                    "Flow Bytes/s": flow["Total Bytes"] / duration if duration > 0 else 0,
                    "Flow Packets/s": flow["Total Packets"] / duration if duration > 0 else 0,
                    "Fwd Packet Length Max": fwd_stats["Max"],
                    "Fwd Packet Length Min": fwd_stats["Min"],
                    "Fwd Packet Length Mean": fwd_stats["Mean"],
                    "Fwd Packet Length Std": fwd_stats["Std"],
                    "Bwd Packet Length Max": bwd_stats["Max"],
                    "Bwd Packet Length Min": bwd_stats["Min"],
                    "Bwd Packet Length Mean": bwd_stats["Mean"],
                    "Bwd Packet Length Std": bwd_stats["Std"],
                    "Packet Length Mean": all_stats["Mean"],
                    "Packet Length Std": all_stats["Std"],
                    "Packet Length Variance": float(np.var(all_lengths, ddof=1)) if len(all_lengths) > 1 else 0,
                    "Min Packet Length": all_stats["Min"],
                    "Max Packet Length": all_stats["Max"],
                    "Average Packet Size": all_stats["Mean"],
                    "Down/Up Ratio": len(bwd) / len(fwd) if len(fwd) else 0,
                    "Flow IAT Mean": calculate_iat(flow["Forward Timestamps"] + flow["Backward Timestamps"])["Mean"],
                    "Flow IAT Std": calculate_iat(flow["Forward Timestamps"] + flow["Backward Timestamps"])["Std"],
                    "Flow IAT Max": calculate_iat(flow["Forward Timestamps"] + flow["Backward Timestamps"])["Max"],
                    "Flow IAT Min": calculate_iat(flow["Forward Timestamps"] + flow["Backward Timestamps"])["Min"],
                    "Fwd IAT Mean": fwd_iat["Mean"],
                    "Fwd IAT Std": fwd_iat["Std"],
                    "Fwd IAT Max": fwd_iat["Max"],
                    "Fwd IAT Min": fwd_iat["Min"],
                    "Bwd IAT Mean": bwd_iat["Mean"],
                    "Bwd IAT Std": bwd_iat["Std"],
                    "Bwd IAT Max": bwd_iat["Max"],
                    "Bwd IAT Min": bwd_iat["Min"],
                    "SYN Flag Count": flow["SYN Flag Count"],
                    "ACK Flag Count": flow["ACK Flag Count"],
                    "FIN Flag Count": flow["FIN Flag Count"],
                    "RST Flag Count": flow["RST Flag Count"],
                    "PSH Flag Count": flow["PSH Flag Count"],
                    "URG Flag Count": flow["URG Flag Count"]
                }

                rows.append(row)

            expired_keys = [
                key for key, flow in flows.items()
                if current_time - flow["Last Seen"] > FLOW_IDLE_TIMEOUT
            ]

            for key in expired_keys:
                del flows[key]

            if rows:

                df = pd.DataFrame(rows)

                df["Start Time"] = pd.to_datetime(df["Start Time"], unit="s")
                df["Last Seen"] = pd.to_datetime(df["Last Seen"], unit="s")

                df.to_csv(OUTPUT_FILE, index=False)

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Exported {len(df)} flows | "
                    f"Expired {len(expired_keys)} idle flows"
                )


print("\n========== NETGUARD AI ENHANCED LIVE EXPORTER ==========")
print("Capturing packet-level statistics...")
print("Export interval:", EXPORT_INTERVAL, "seconds")
print("Flow idle timeout:", FLOW_IDLE_TIMEOUT, "seconds")
print("Output file:", OUTPUT_FILE)
print("Press CTRL+C to stop.\n")


threading.Thread(
    target=export_flows,
    daemon=True
).start()


try:

    sniff(
        prn=process_packet,
        store=False
    )

except KeyboardInterrupt:

    print("\nEnhanced live capture stopped.")
