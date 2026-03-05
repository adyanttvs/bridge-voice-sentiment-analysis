import pandas as pd

excel_path = r"C:\Users\k02401\Downloads\outbound_call_recording_1772183015978.xlsx"
df = pd.read_excel(excel_path)
print("Columns:", df.columns.tolist())
print("First 3 rows:")
print(df.head(3).to_markdown())
