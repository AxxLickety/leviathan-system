import pandas as pd
import matplotlib.pyplot as plt

file = "Copy of MAT133 数据.xlsx"

years = ["1994", "2004", "2014", "2024"]

gini_values = []
lorenz_data = {}

for year in years:
    df = pd.read_excel(file, sheet_name=year)
    
    # Lorenz curve
    P = df["P"]
    L = df["Share"]
    
    lorenz_data[year] = (P, L)
    
    # Gini calculation
    area = df["TrapArea"].sum()
    gini = 1 - 2 * area
    gini_values.append(gini)

# -------- Gini vs Time --------
plt.figure()
plt.plot(years, gini_values)
plt.xlabel("Year")
plt.ylabel("Gini Coefficient")
plt.title("Gini Coefficient for Austria Over Time")
plt.savefig("gini_over_time.png")
plt.close()

# -------- Lorenz Curves --------
plt.figure()

for year in years:
    P, L = lorenz_data[year]
    plt.plot(P, L, label=year)

# equality line
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Population Share")
plt.ylabel("Income Share")
plt.title("Lorenz Curves for Austria")
plt.legend()
plt.savefig("lorenz_curves.png")
plt.close()