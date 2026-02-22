import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

# 1. global regions
regions = pd.DataFrame({
    "city": ["Austin", "Toronto", "Shanghai", "Hong Kong", "London", "Tokyo", "Singapore"],
    "slug": ["austin", "toronto", "shanghai", "hong_kong", "london", "tokyo", "singapore"],
    "country": ["USA", "Canada", "China", "Hong Kong", "UK", "Japan", "Singapore"],
    "regime": ["us_elastic", "ca_restrict", "cn_admin", "hk_supply_lock", "uk_greenbelt", "jp_supply", "sg_hdb"],
})
regions.to_csv("data/global_regions.csv", index=False)

# 2. housing price
price = pd.DataFrame({
    "region": ["austin", "toronto", "shanghai", "hong_kong"],
    "date": ["2015-01-01"]*4,
    "price": [250000, 520000, 28000, 95000]
})
price.to_csv("data/raw/housing_price.csv", index=False)

# 3. income
income = pd.DataFrame({
    "region": ["austin", "toronto", "shanghai", "hong_kong"],
    "date": ["2015-01-01"]*4,
    "income": [60000, 72000, 150000, 200000]
})
income.to_csv("data/raw/income.csv", index=False)

# 4. mortgage rates
rates = pd.DataFrame({
    "date": ["2015-01-01"]*4,
    "country": ["USA", "Canada", "China", "Hong Kong"],
    "mortgage_rate": [0.045, 0.028, 0.050, 0.025]
})
rates.to_csv("data/raw/mortgage_rate.csv", index=False)

# 5. population migration
pop = pd.DataFrame({
    "region": ["austin", "toronto", "shanghai", "hong_kong"],
    "date": ["2015-01-01"]*4,
    "population": [1800000, 6000000, 24000000, 7500000],
    "net_migration": [3000, 3500, 5000, -2000]
})
pop.to_csv("data/raw/population_migration.csv", index=False)

print("All mock CSVs generated!")

import os

print("Script is running!")

os.makedirs("data", exist_ok=True)

with open("data/test_output.txt", "w") as f:
    f.write("Hello Axl!")

print("All mock CSVs generated!")
