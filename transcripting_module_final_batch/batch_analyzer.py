import os
import time
import re
import pandas as pd
import httpx
from app_final import process_audio_file

# Define paths
excel_path = r"C:\Users\k02401\Downloads\outbound_call_recording_1772183015978.xlsx"
output_path = r"C:\Users\k02401\Downloads\batch_analysis_results.csv"

# Load the file
df = pd.read_excel(excel_path)

# Filter for length > 100
filtered_df = df[df['Length In Sec'] > 100].copy()
print(f"Found {len(filtered_df)} records with length > 100 seconds.")

results = []
total_processed_seconds = 0

for idx, row in filtered_df.iterrows():
    phone = row.get('Phone Number', 'Unknown')
    url = row.get('Location', '')
    length = row.get('Length In Sec', 0)
    
    if pd.isna(url) or not str(url).startswith('http'):
        continue
        
    print(f"\n[{idx}] Processing Phone: {phone} (Length: {length}s)")
    print(f"URL: {url}")
    
    if total_processed_seconds + length > 3500:
        print(f"-> Reached 1-hour audio limit ({total_processed_seconds}s + {length}s > 3500s). Stopping batch.")
        break
        
    
    temp_filename = f"batch_temp_{int(time.time())}_{idx}.mp3"
    
    try:
        # Download
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            with open(temp_filename, "wb") as f:
                f.write(resp.content)
                
        # Process with retry for Rate Limit
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = process_audio_file(temp_filename, "auto")
                audit = data.get('audit', {})
                
                results.append({
                    'Phone Number': phone,
                    'Length In Sec': length,
                    'Audio URL': url,
                    'Total Score': audit.get('total_score', 'Error'),
                    'Risk Flags': ', '.join(audit.get('risk_flags', [])),
                    'Sentiment Label': audit.get('sentiment', {}).get('label', 'Unknown'),
                    'Summary': audit.get('summary', ''),
                    'Detected Language': data.get('detected_language_code', 'Unknown')
                })
                
                print(f"-> Score: {audit.get('total_score')} | Sentiment: {audit.get('sentiment', {}).get('label')}")
                total_processed_seconds += length
                break # Success, exit retry loop
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'rate_limit_exceeded' in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 65
                        match = re.search(r'try again in (?:(\d+)m)?([\d\.]+)s', error_msg)
                        if match:
                            mins = int(match.group(1)) if match.group(1) else 0
                            secs = float(match.group(2))
                            wait_time = mins * 60 + secs + 5
                        print(f"-> Rate limit hit for {phone}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise Exception("Max retries reached due to rate limit: " + error_msg)
                elif 'timeout' in error_msg.lower() or 'read' in error_msg.lower() or 'connection' in error_msg.lower():
                    print(f"-> Network/Timeout Error for {phone}: {error_msg}. Retrying {attempt + 1}/{max_retries}...")
                    time.sleep(10)
                    if attempt == max_retries - 1:
                        raise Exception("Max retries reached due to network/timeout error: " + error_msg)
                else:
                    raise e # Re-raise non-rate-limit errors
                    
    except Exception as e:
        print(f"-> Error processing {phone}: {e}")
        results.append({
            'Phone Number': phone,
            'Length In Sec': length,
            'Audio URL': url,
            'Total Score': 'Error',
            'Summary': str(e)
        })
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"\nBatch processing complete. Results saved to {output_path}")
