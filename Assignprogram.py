import matplotlib.pyplot as plt

# Define data points
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create the plot
plt.plot(x, y, marker='o', color='b', linestyle='-')

# Add titles and labels
plt.title("Simple Line Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# Display the plot
plt.show()
