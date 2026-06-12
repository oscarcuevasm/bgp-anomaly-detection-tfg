# ==============================================================================
# Copyright (c) UNIVERSIDAD AUTÓNOMA DE MADRID
# Francisco Tomás y Valiente, no 1
# Madrid, 28049
# Spain
#
# Óscar Cuevas Martínez
# Evaluating the Performance of BGP Different Anomaly Detection Methods
# All Rights Reserved
# ==============================================================================



import os
import gzip
import shutil
import subprocess
import csv
from datetime import datetime, timedelta
from urllib.request import urlopen
from pathlib import Path

# --- PATH CONFIGURATION ---
BASE_URL = "https://data.ris.ripe.net/rrc04/2017.05"
RIPE_DIR = "./Wannacrypt_RRC04_Raw"
OUTPUT_DIR = os.path.join(RIPE_DIR, "mrt_files")
TEMP_DIR = os.path.join(RIPE_DIR, "temp_mrt")

# --- TIME RANGES AND OUTPUT FILE NAMES ---
# (Start, End, Output_File)
PERIODS = [
    ("20170513.0000", "20170513.2355", os.path.join(RIPE_DIR, "wannacrypt_13may_raw.csv")),
    ("20170506.0000", "20170506.2355", os.path.join(RIPE_DIR, "wannacrypt_baseline_6may_raw.csv"))
]

fieldnames = ['MRT_Type', 'Time', 'Entry_Type', 'Peer_IP', 'Peer_AS',
              'Prefix', 'AS_Path', 'Origin', 'Next_Hop', 'Local_Pref',
              'MED', 'Community', 'Atomic_Aggregate', 'Aggregator', 'Label']

def create_directories():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

def download_file(url, local_path):
    try:
        with urlopen(url) as response:
            with open(local_path, 'wb') as out_file:
                out_file.write(response.read())
        return True
    except Exception:
        return False

def decompress_gz(gz_file, output_file):
    try:
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"✗ Error decompressing {gz_file}: {e}")
        return False

# --- BGP EXTRACTION LOGIC ---
def parse_bgpdump_line(line):
    line = line.strip()
    if not line: return None
    parts = line.split('|')
    if len(parts) < 6: return None

    try:
        msg_type = parts[0]
        if msg_type != 'BGP4MP': return None

        timestamp = int(parts[1])
        dt = datetime.utcfromtimestamp(timestamp)
        date_time = dt.strftime('%Y-%m-%d %H:%M:%S')

        update_type = parts[2]
        peer_ip = parts[3]
        peer_as = parts[4]
        prefix = parts[5]

        if update_type == 'W':
            return {
                'MRT_Type': 'BGP4MP', 'Time': date_time, 'Entry_Type': 'W',
                'Peer_IP': peer_ip, 'Peer_AS': peer_as, 'Prefix': prefix,
                'AS_Path': '', 'Origin': '', 'Next_Hop': '', 'Local_Pref': '',
                'MED': '', 'Community': '', 'Atomic_Aggregate': '', 'Aggregator': '',
                'Label': 'normal'
            }

        as_path = parts[6] if len(parts) > 6 else ''
        origin = parts[7] if len(parts) > 7 else ''
        next_hop = parts[8] if len(parts) > 8 else ''
        local_pref = parts[9] if len(parts) > 9 else ''
        med = parts[10] if len(parts) > 10 else ''
        community = parts[11] if len(parts) > 11 else ''
        atomic_agg = parts[12] if len(parts) > 12 else ''
        aggregator = parts[13] if len(parts) > 13 else ''

        return {
            'MRT_Type': 'BGP4MP', 'Time': date_time, 'Entry_Type': 'A',
            'Peer_IP': peer_ip, 'Peer_AS': peer_as, 'Prefix': prefix,
            'AS_Path': as_path, 'Origin': origin, 'Next_Hop': next_hop,
            'Local_Pref': local_pref, 'MED': med, 'Community': community,
            'Atomic_Aggregate': atomic_agg, 'Aggregator': aggregator,
            'Label': 'normal'
        }
    except (ValueError, IndexError):
        return None

def parse_mrt_file_with_bgpdump(mrt_file):
    records = []
    try:
        result = subprocess.run(['bgpdump', '-m', mrt_file], capture_output=True, text=True, check=False)
        if result.returncode != 0: return []
        lines = result.stdout.strip().split('\n')
        for line in lines:
            record = parse_bgpdump_line(line)
            if record: records.append(record)
    except Exception:
        pass
    return records

def collect_and_process_updates():
    create_directories()
    
    for start_str, end_str, csv_output in PERIODS:
        print("\n" + "=" * 70)
        print(f"PROCESANDO PERIODO: {start_str[:8]} -> Guardando en {os.path.basename(csv_output)}")
        print("=" * 70)

        # 1. Identify files to download
        files_to_download = []
        start_time = datetime.strptime(start_str, "%Y%m%d.%H%M")
        end_time = datetime.strptime(end_str, "%Y%m%d.%H%M")

        current_time = start_time
        while current_time <= end_time:
            filename = f"updates.{current_time.strftime('%Y%m%d.%H%M')}.gz"
            files_to_download.append(filename)
            current_time += timedelta(minutes=5)

        # 2. Download files
        downloaded_files = []
        for i, filename in enumerate(files_to_download, 1):
            url = f"{BASE_URL}/{filename}"
            local_path = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(local_path):
                downloaded_files.append(local_path)
                continue

            print(f"[{i}/{len(files_to_download)}] Downloading {filename}...")
            if download_file(url, local_path):
                downloaded_files.append(local_path)
            else:
                print(f"  -> {filename} not found.")

        # 3. Extract and write to the corresponding CSV
        with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            writer.writeheader()

        total_records = 0
        total_announcements = 0
        total_withdrawals = 0

        for i, gz_file in enumerate(downloaded_files, 1):
            mrt_file = os.path.join(TEMP_DIR, os.path.basename(gz_file).replace('.gz', ''))
            print(f"[{i}/{len(downloaded_files)}] Procesando {os.path.basename(gz_file)}...", end=" ")

            if not decompress_gz(gz_file, mrt_file):
                print("Failed")
                continue

            records = parse_mrt_file_with_bgpdump(mrt_file)
            
            if records:
                with open(csv_output, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
                    writer.writerows(records)
                    
                total_records += len(records)
                total_announcements += sum(1 for r in records if r['Entry_Type'] == 'A')
                total_withdrawals += sum(1 for r in records if r['Entry_Type'] == 'W')
                print(f"✓ {len(records)} registros")
            else:
                print("✓ 0 registros")

            try:
                os.remove(mrt_file)
            except:
                pass

        print(f"\nSummary for {os.path.basename(csv_output)}:")
        print(f"✓ Total packets: {total_records:,} (A: {total_announcements:,} | W: {total_withdrawals:,})")

    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

if __name__ == "__main__":
    collect_and_process_updates()