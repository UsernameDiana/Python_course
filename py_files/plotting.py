
import matplotlib.pyplot as plt

squares = [x ** 2 for x in range(1, 10)]

plt.plot(squares)

plt.show()

print(squares)

# making lables and line thickness for grapg
squares = [x ** 2 for x in range(1, 6)]

plt.plot(squares, linewidth=5)
# Set chart title and label axes.
plt.title("Square Numbers", fontsize=24)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Square of Value", fontsize=14)
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)
plt.show()


# defining colors
x_values = range(1, 1001)
y_values = [x ** 2 for x in x_values]

plt.scatter(x_values, y_values, c=y_values, 
            cmap=plt.cm.Blues, edgecolor='none', s=40)
plt.axis([0, 1100, 0, 1100000])


# dot plotting
x_values = range(1, 6)
y_values = [x ** 2 for x in x_values]

print('Points to be plotted {}'.format(list(zip(x_values, y_values))))

plt.scatter(x_values, y_values, s=100)