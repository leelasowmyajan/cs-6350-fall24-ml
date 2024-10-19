import pandas as pd
import matplotlib.pyplot as plt
import utilities

def plot_data(result_csv, title, plot_filename):
    df = pd.read_csv(result_csv)

    # Plot the cost function over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(df['Iterations'], df['Cost'], linestyle='-', color='b', label='Costs')  
    plt.title(title, fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(utilities.results_path + plot_filename, format='png')

    # Display the plot
    plt.show()

def main():
    plot_data(utilities.results_bgd_csv, 'Cost Function Over Iterations (Batch Gradient Descent)', 'bgd_plot.png')
    #plot_data(utilities.results_sgd_csv, 'Cost Function Over Iterations (Stochastic Gradient Descent)', 'sgd_plot.png')

if __name__ == "__main__":
    main()
