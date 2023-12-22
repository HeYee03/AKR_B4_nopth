#####################没用上###############

import matplotlib.pyplot as plt

# Plotting loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Plotting accuracy
plt.figure(figsize=(10, 5))
plt.plot(accuracies)
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
