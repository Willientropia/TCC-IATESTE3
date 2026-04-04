import json
d = json.load(open("dados/training_data_real.json", "r", encoding="utf-8"))
grid_nonzero = sum(1 for day in d["days"] for v in day["grid"] if abs(v) > 1)
total = len(d["days"]) * 96
print(f"Grid nao-zero: {grid_nonzero} slots de {total} total")
# Checar o campo Grid Power no CSV original
import csv
with open("REALTIME_REPORT_ROW_J05H502051HB122_30.csv", "r", encoding="utf-8") as f:
    f.readline()  # skip SN line
    reader = csv.DictReader(f)
    count = 0
    nonzero = 0
    for row in reader:
        gp = row.get("Grid Power/kW", "").strip()
        count += 1
        if gp and gp != "" and gp != "0.0" and gp != "0":
            nonzero += 1
            if nonzero <= 5:
                print(f"  Grid Power nao-zero: {row['time']} = {gp}")
    print(f"Total rows: {count}, Grid nao-zero: {nonzero}")
