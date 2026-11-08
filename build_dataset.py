"""Build raw-byte, pooled-context wavelet, and graph metadata for Traffic-Diff.

Each .pcap filename is treated as a class label, matching the user's existing pipeline.
Output arrays:
  raw_bytes: (N, MAX_PACKETS, PACKET_BYTES), normalized to [0, 1]
  wavelet:   (N, 1, SCALES, MAX_PACKETS), pooled context scalogram
  label:     (N,)
  src_id/dst_id/start_time/end_time/capture_id: graph metadata for FIGR

Requires: pip install scapy PyWavelets numpy
"""
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pywt
from scapy.all import PcapReader, IP, TCP, UDP

from config import cfg

SEED = cfg.SEED
random.seed(SEED)
np.random.seed(SEED)


def flow_key(pkt):
    if not pkt.haslayer(IP):
        return None
    ip = pkt[IP]
    if pkt.haslayer(TCP):
        sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
    elif pkt.haslayer(UDP):
        sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
    else:
        return None
    proto = int(ip.proto)
    a = (str(ip.src), sport, str(ip.dst), dport, proto)
    b = (str(ip.dst), dport, str(ip.src), sport, proto)
    return min(a, b)


def cwt_scalogram(seq):
    seq = np.asarray(seq, dtype=np.float32)
    if seq.size != cfg.MAX_PACKETS:
        fixed = np.zeros(cfg.MAX_PACKETS, dtype=np.float32)
        fixed[:min(len(seq), cfg.MAX_PACKETS)] = seq[:cfg.MAX_PACKETS]
        seq = fixed
    scales = np.arange(1, cfg.SCALES + 1)
    try:
        coef, _ = pywt.cwt(seq, scales, cfg.WAVELET)
        power = np.log1p(np.abs(coef) ** 2)
        return np.tanh(power / 4.0).astype(np.float32)
    except Exception as exc:
        print(f"[WARN] CWT failed, using zero scalogram: {exc}")
        return np.zeros((cfg.SCALES, cfg.MAX_PACKETS), dtype=np.float32)


def packet_bytes(pkt):
    # IP bytes only; no Ethernet header. Clip/pad each packet to fixed byte length.
    try:
        data = bytes(pkt[IP])
    except Exception:
        data = bytes(pkt)
    arr = np.zeros(cfg.PACKET_BYTES, dtype=np.float32)
    n = min(len(data), cfg.PACKET_BYTES)
    if n:
        arr[:n] = np.frombuffer(data[:n], dtype=np.uint8).astype(np.float32) / 255.0
    return arr


def extract_pcap(path, capture_id, ip_to_id):
    flows = defaultdict(list)
    meta = {}
    packet_count = 0
    valid_seen = 0
    with PcapReader(str(path)) as reader:
        for pkt in reader:
            packet_count += 1
            if packet_count > cfg.MAX_PACKETS_PER_PCAP:
                print(f"[WARN] packet safety limit reached for {path.name}")
                break
            key = flow_key(pkt)
            if key is None:
                continue
            ts = float(pkt.time)
            ip = pkt[IP]
            src, dst = str(ip.src), str(ip.dst)
            if src not in ip_to_id:
                ip_to_id[src] = len(ip_to_id)
            if dst not in ip_to_id:
                ip_to_id[dst] = len(ip_to_id)
            if key not in meta:
                meta[key] = {
                    "src_ip": src, "dst_ip": dst,
                    "src_id": ip_to_id[src], "dst_id": ip_to_id[dst],
                    "start": ts, "end": ts,
                }
            meta[key]["start"] = min(meta[key]["start"], ts)
            meta[key]["end"] = max(meta[key]["end"], ts)
            flows[key].append((ts, packet_bytes(pkt), len(bytes(pkt[IP]))))
            
            ###########################################################################################
            # if len(flows[key]) == cfg.MIN_PACKETS:
            #     valid_seen += 1
            #     if valid_seen >= 1000:
            #         print(f"[INFO] reached {valid_seen} valid flows, stopping early for {path.name}")
            #         break
            ##############################################################################################

    keys = [k for k, ps in flows.items() if len(ps) >= cfg.MIN_PACKETS]
    # Build pooled context from packet-length sequences of flows sharing endpoints or
    # overlapping the target's temporal neighborhood. No labels are used in context.
    records = []
    for key in keys:
        pkts = sorted(flows[key], key=lambda x: x[0])
        target = meta[key]
        raw = np.zeros((cfg.MAX_PACKETS, cfg.PACKET_BYTES), dtype=np.float32)
        n = min(len(pkts), cfg.MAX_PACKETS)
        raw[:n] = np.stack([p[1] for p in pkts[:n]])
        # Context sequence: average normalized packet lengths from related flows.
        left = target["start"] - cfg.CONTEXT_WINDOW
        right = target["end"] + cfg.CONTEXT_WINDOW
        ctx_candidates = []
        for other_key in keys:
            if other_key == key:
                continue
            om = meta[other_key]
            same_endpoint = (om["src_ip"] == target["src_ip"] or
                             om["dst_ip"] == target["dst_ip"] or
                             om["src_ip"] == target["dst_ip"] or
                             om["dst_ip"] == target["src_ip"])
            time_near = om["start"] <= right and om["end"] >= left
            if same_endpoint or time_near:
                ctx_candidates.append(other_key)
        # Bound the local context for memory/runtime stability.
        ctx_candidates.sort(key=lambda k: abs(meta[k]["start"] - target["start"]))
        ctx_candidates = ctx_candidates[:64]
        if ctx_candidates:
            seqs = []
            for ck in ctx_candidates:
                cp = sorted(flows[ck], key=lambda x: x[0])[:cfg.MAX_PACKETS]
                seq = np.zeros(cfg.MAX_PACKETS, dtype=np.float32)
                for j, p in enumerate(cp):
                    seq[j] = min(float(p[2]) / 1500.0, 1.0)
                seqs.append(seq)
            context_seq = np.mean(np.stack(seqs), axis=0).astype(np.float32)
        else:
            context_seq = np.zeros(cfg.MAX_PACKETS, dtype=np.float32)
        # Ensure context has a valid signal even when the local neighborhood is empty.
        if not np.any(context_seq):
            for j, p in enumerate(pkts[:cfg.MAX_PACKETS]):
                context_seq[j] = min(float(p[2]) / 1500.0, 1.0)
        wave = cwt_scalogram(context_seq)[None, :, :]  # (1, scales, packets)
        records.append({
            "raw_bytes": raw,
            "wavelet": wave,
            "src_id": target["src_id"],
            "dst_id": target["dst_id"],
            "start_time": target["start"],
            "end_time": target["end"],
            "capture_id": capture_id,
        })
    return records


def build_dataset():
    root = Path(cfg.DENOISE_DIR)
    if not root.exists():
        raise FileNotFoundError(f"PCAP directory not found: {root.resolve()}")
    pcap_files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pcap")
    if not pcap_files:
        raise FileNotFoundError(f"No .pcap files found in {root.resolve()}")

    class_names = [p.stem for p in pcap_files]
    all_records, all_labels = [], []
    ip_to_id = {}
    for capture_id, path in enumerate(pcap_files):
        print(f"\n[PCAP] {path.name} -> class {capture_id}")
        records = extract_pcap(path, capture_id, ip_to_id)
        print(f"  valid flows: {len(records)}")
        # if len(records) > cfg.MAX_FLOWS_PER_CLASS:
        #     records = random.sample(records, cfg.MAX_FLOWS_PER_CLASS)
        #     print(f"  sampled to: {len(records)}")
        all_records.extend(records)
        all_labels.extend([capture_id] * len(records))

    if not all_records:
        raise RuntimeError("No valid flows extracted. Check PCAP files and MIN_PACKETS.")
    data = {
        "raw_bytes": np.stack([r["raw_bytes"] for r in all_records]).astype(np.float32),
        "wavelet": np.stack([r["wavelet"] for r in all_records]).astype(np.float32),
        "label": np.asarray(all_labels, dtype=np.int64),
        "src_id": np.asarray([r["src_id"] for r in all_records], dtype=np.int64),
        "dst_id": np.asarray([r["dst_id"] for r in all_records], dtype=np.int64),
        "start_time": np.asarray([r["start_time"] for r in all_records], dtype=np.float32),
        "end_time": np.asarray([r["end_time"] for r in all_records], dtype=np.float32),
        "capture_id": np.asarray([r["capture_id"] for r in all_records], dtype=np.int64),
    }
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    out_path = Path(cfg.DATA_DIR) / cfg.DATA_FILE
    np.savez_compressed(out_path, **data)
    with open(Path(cfg.DATA_DIR) / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")
    print("\n========== Dataset summary ==========")
    for key, value in data.items():
        print(f"{key:12s}: {value.shape}")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    build_dataset()

