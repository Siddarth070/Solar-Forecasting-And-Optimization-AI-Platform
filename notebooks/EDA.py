import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  


df = pd.read_parquet('data/processed/jaipur_simulated_90d.parquet')

print("Max output:", df['solar_output_mw'].max())
print("Min output:", df['solar_output_mw'].min())

daytime = df[df['is_daytime'] == True]

plt.scatter(daytime['cloud_cover'], 
            daytime['solar_output_mw'], 
            alpha=0.2, s=5)
plt.title('Solar output vs cloud cover — daytime only')
plt.xlabel('Cloud cover (%)')
plt.ylabel('Solar output (MW)')
plt.grid(True)
plt.show()
