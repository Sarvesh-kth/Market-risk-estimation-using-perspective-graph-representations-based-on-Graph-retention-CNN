import pandas as pd
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt


sector_data = pd.read_excel('Results.xlsx')

sector_data['Difference'] = (sector_data['Predicted'] - sector_data['Actual']).abs()
sector_data['Score'] = 1 - 0.25 * sector_data['Difference']


sector_performance = sector_data.groupby('Sector Group')['Score'].mean()
tsp = sector_data['Score'].mean()

sector_data['Difference1'] = (sector_data['Predicted'] - sector_data['Actual'])^2

sector_data['Squared Difference'] = (sector_data['Predicted'] - sector_data['Actual'])**2
sector_data['Score1'] = 1 - (sector_data['Squared Difference'] / 16)

updated_sector_performance = sector_data.groupby('Sector Group')['Score1'].mean()
utsp = sector_data['Score1'].mean()

print(sector_performance)
print(tsp)

print(updated_sector_performance)
print(utsp)

roc_info = {}
classes = np.unique(sector_data['Actual'])

for cls in classes:

    actual = (sector_data['Actual'] == cls).astype(int)
    predicted_prob = (sector_data['Predicted'] == cls).astype(int)  

    fpr, tpr, thresholds = roc_curve(actual, predicted_prob)
    roc_auc = auc(fpr, tpr)


    roc_info[cls] = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'AUC': roc_auc
    }


for cls in classes:
    print(f"Class {cls} AUC: ", roc_info[cls]['AUC'])

fig, ax = plt.subplots(figsize=(10, 8))
for cls in classes:
    fpr, tpr = roc_info[cls]['fpr'], roc_info[cls]['tpr']
    roc_auc = roc_info[cls]['AUC']
    ax.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves for Each Class')
ax.legend(loc='lower right')
plt.show()
