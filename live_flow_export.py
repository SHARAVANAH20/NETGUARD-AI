
from scapy.all import sniff, IP, TCP, UDP
from pathlib import Path
from datetime import datetime
import pandas as pd
import threading
import time


OUTPUT_DIR = Path("data/live")
OUTPUT_FILE = OUTPUT_DIR / "live_flows.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_INTERVAL = 10
FLOW_IDLE_TIMEOUT = 60

flows = {}
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

            flows[flow_key] = {
                "Source IP": src_ip,
                "Destination IP": dst_ip,
                "Source Port": src_port,
                "Destination Port": dst_port,
                "Protocol": protocol,
                "Start Time": current_time,
                "Last Seen": current_time,
                "Total Packets": 0,
                "Total Bytes": 0,
                "Forward Packets": 0,
                "Backward Packets": 0
            }

        flow = flows[flow_key]

        flow["Last Seen"] = current_time
        flow["Total Packets"] += 1
        flow["Total Bytes"] += packet_size

        if direction == "forward":
            flow["Forward Packets"] += 1
        else:
            flow["Backward Packets"] += 1


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

                row = flow.copy()

                row["Duration Seconds"] = duration

                row["Average Packet Size"] = (
                    flow["Total Bytes"] / flow["Total Packets"]
                    if flow["Total Packets"] > 0
                    else 0
                )

                rows.append(row)

            # Remove flows inactive for more than 60 seconds
            expired_keys = [
                key
                for key, flow in flows.items()
                if current_time - flow["Last Seen"] > FLOW_IDLE_TIMEOUT
            ]

            for key in expired_keys:
                del flows[key]

            if rows:

                df = pd.DataFrame(rows)

                df["Start Time"] = pd.to_datetime(
                    df["Start Time"],
                    unit="s"
                )

                df["Last Seen"] = pd.to_datetime(
                    df["Last Seen"],
                    unit="s"
                )

                df.to_csv(OUTPUT_FILE, index=False)

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Exported {len(df)} flows | "
                    f"Expired {len(expired_keys)} idle flows"
                )

            else:
                print("No live flows currently available.")


print("\n========== NETGUARD AI LIVE FLOW EXPORTER ==========")
print("Capturing TCP and UDP traffic...")
print(f"Export interval: {EXPORT_INTERVAL} seconds")
print(f"Flow idle timeout: {FLOW_IDLE_TIMEOUT} seconds")
print(f"Output file: {OUTPUT_FILE}")
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

    print("\nLive capture stopped.")
