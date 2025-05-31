import pandas as pd
import matplotlib.pyplot as plt

# --- Figure 1: Bar chart of top-5 test acc after 5 epochs ---
df_initial = pd.DataFrame({
    'Config': [
        'Adam,1e-4,32,BN',
        'SGD,1e-2,64,BN',
        'SGD,1e-2,128,BN',
        'SGD,1e-3,32,BN',
        'Adam,1e-4,64,BN',
    ],
    'Acc@5': [0.7907, 0.7802, 0.7688, 0.7675, 0.7669]
})
plt.figure()
plt.bar(df_initial['Config'], df_initial['Acc@5'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Test Accuracy @ Epoch 5')
plt.title('Fig. 1. Top-5 Configurations After 5 Epochs')
plt.tight_layout()
plt.savefig('fig1_top5_bar.png')

# --- Figure 2: Line chart of acc every 5 epochs up to 30 ---
df_long = pd.DataFrame({
    'epoch': [5,10,15,20,25,30],
    'Adam,1e-4,32,BN': [0.782,0.8242,0.851,0.8582,0.8681,0.8709],
    'SGD,1e-2,64,BN': [0.7882,0.8067,0.8576,0.8635,0.8716,0.8813],
    'SGD,1e-2,128,BN':[0.7873,0.8137,0.847,0.8263,0.8685,0.8730],
    'SGD,1e-3,32,BN': [0.7709,0.8097,0.8503,0.8542,0.8495,0.8627],
    'Adam,1e-4,64,BN':[0.7507,0.8169,0.8291,0.8466,0.8608,0.8535],
})
plt.figure()
for cfg in df_long.columns[1:]:
    plt.plot(df_long['epoch'], df_long[cfg], marker='o', label=cfg)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Fig. 2. Test Accuracy Over 30 Epochs (Every 5)')
plt.legend(fontsize='small')
plt.tight_layout()
plt.savefig('fig2_longrun.png')
