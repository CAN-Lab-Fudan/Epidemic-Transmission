import math
import matplotlib.pyplot as plt
from mpmath import ln
import numpy as np

def compute_w_values(x, r):
    """
    Compute w(s) based on the given x values and parameter r.
    The function uses the formula: w(s) = exp(-(-ln(s))**r)

    Parameters:
    - x: List of values of s
    - r: Parameter to modify the formula

    Returns:
    - List of computed w(s) values
    """
    return [math.exp(-(-ln(l)) ** r) for l in x]


def plot_w_values(x, w1, w2, w, output_filename="WE.png"):
    """
    Plot the values of w(s) for different values of r and save the plot.

    Parameters:
    - x: List of s values
    - w1, w2, w: Lists of w(s) values for different r
    - output_filename: Name of the output file where the plot will be saved
    """
    # Plot the w(s) values for each r
    plt.plot(x, w1, label=r'$\gamma=0.3$', linestyle='-', marker='d', markersize=8, linewidth=2, color='r')
    plt.plot(x, w2, label=r'$\gamma=0.5$', linestyle='-', marker='X', markersize=8, linewidth=2, color='b')
    plt.plot(x, w, label=r'$\gamma=1$', linestyle='-', linewidth=2, color='black', mfc='w')

    # Label and customize the plot
    plt.tick_params(labelsize=15)
    plt.xlabel('s', fontsize=20)
    plt.ylabel(r'$w(s)$', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(ls='--')  # Add grid

    # Save the plot to a file
    plt.savefig(output_filename)

    # Annotate the minimum value
    min_w_value = min(w)
    min_index = w.index(min_w_value)
    plt.annotate('Minimum', (x[min_index], min_w_value), (x[min_index] + 0.5, min_w_value),
                 xycoords='data', arrowprops=dict(facecolor='r', shrink=0.1), c='r', fontsize=15)

    # Show the plot
    plt.show()


def main():
    # Define s values
    # x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    x = np.linspace(0, 1, 21)  # 21 points from 0 to 1, inclusive
    print(x)
    # Compute w(s) values for different gamma (r values)
    r_values = [1, 0.3, 0.5]
    w_values = [compute_w_values(x, r) for r in r_values]

    # Plot the results
    # plot_w_values(x, w_values[1], w_values[2], w_values[0])


if __name__ == "__main__":
    main()
