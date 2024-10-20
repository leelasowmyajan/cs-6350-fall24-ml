import pandas as pd
import matplotlib.pyplot as plt
import utilities

def plot_bagging_data(result_csv, plot_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(result_csv)

    # Plot both training and testing errors over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(df['Iteration'], df['Training Error (%)'], linestyle='-', color='b', label='Training Error (%)')  
    plt.plot(df['Iteration'], df['Testing Error (%)'], linestyle='-', color='r', label='Testing Error (%)')  

    # Add title and labels
    plt.title('Bagging - Training and Testing Errors Over Iterations', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Error (%)', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(utilities.results_path + plot_filename, format='png')

    # Display the plot
    plt.show()

def plot_adaboost_data(result_csv, plot_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(result_csv)

    # Plot both training and testing errors over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(df['Iteration'], df['Training Error (%)'], linestyle='-', color='b', label='Training Error (%)')  
    plt.plot(df['Iteration'], df['Testing Error (%)'], linestyle='-', color='r', label='Testing Error (%)')  

    # Add title and labels
    plt.title('AdaBoost - Training and Testing Errors Over Iterations', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Error (%)', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(utilities.results_path + plot_filename, format='png')

    # Display the plot
    plt.show()

def plot_stump_data(result_csv, plot_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(result_csv)

    # Plot both training and testing errors over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(df['Iteration'], df['Training Stump Error (%)'], linestyle='-', color='b', label='Training Error (%)')  
    plt.plot(df['Iteration'], df['Testing Stump Error (%)'], linestyle='-', color='r', label='Testing Error (%)')  

    # Add title and labels
    plt.title('Decision Tree Stump - Training and Testing Errors Over Iterations', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Error (%)', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(utilities.results_path + plot_filename, format='png')

    # Display the plot
    plt.show()

def plot_forest_train_data(result_csv, plot_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(result_csv)

    # Plot both training and testing errors over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(df['Iteration'], df['2 Features Training Error (%)'], linestyle='-', color='b', label='2 Features Training Error (%)')  
    plt.plot(df['Iteration'], df['4 Features Training Error (%)'], linestyle='-', color='r', label='4 Features Training Error (%)')  
    plt.plot(df['Iteration'], df['6 Features Training Error (%)'], linestyle='-', color='g', label='6 Features Training Error (%)')  

    # Add title and labels
    plt.title('Random Forest - Training Errors Over Iterations', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Training Error (%)', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(utilities.results_path + plot_filename, format='png')

    # Display the plot
    plt.show()

def plot_forest_test_data(result_csv, plot_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(result_csv)

    # Plot both training and testing errors over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(df['Iteration'], df['2 Features Testing Error (%)'], linestyle='-', color='b', label='2 Features Testing Error (%)')  
    plt.plot(df['Iteration'], df['4 Features Testing Error (%)'], linestyle='-', color='r', label='4 Features Testing Error (%)')  
    plt.plot(df['Iteration'], df['6 Features Testing Error (%)'], linestyle='-', color='g', label='6 Features Testing Error (%)')  

    # Add title and labels
    plt.title('Random Forest - Testing Errors Over Iterations', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Testing Error (%)', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(utilities.results_path + plot_filename, format='png')

    # Display the plot
    plt.show()

def plot_forest_train_test_data(result_csv, plot_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(result_csv)

    # Create the figure and axis
    plt.figure(figsize=(10, 7))

    # Plot training errors
    plt.plot(df['Iteration'], df['2 Features Training Error (%)'], linestyle='-', color='b', label='2 Features Training Error (%)')  
    plt.plot(df['Iteration'], df['4 Features Training Error (%)'], linestyle='-', color='r', label='4 Features Training Error (%)')  
    plt.plot(df['Iteration'], df['6 Features Training Error (%)'], linestyle='-', color='g', label='6 Features Training Error (%)')  

    # Plot testing errors
    plt.plot(df['Iteration'], df['2 Features Testing Error (%)'], linestyle='--', color='b', label='2 Features Testing Error (%)')  
    plt.plot(df['Iteration'], df['4 Features Testing Error (%)'], linestyle='--', color='r', label='4 Features Testing Error (%)')  
    plt.plot(df['Iteration'], df['6 Features Testing Error (%)'], linestyle='--', color='g', label='6 Features Testing Error (%)')  

    # Add title and labels
    plt.title('Random Forest - Training and Testing Errors Over Iterations', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Error (%)', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(utilities.results_path + plot_filename, format='png')

    # Display the plot
    plt.show()

def main():
    plot_adaboost_data(utilities.results_adaboost_csv, 'adaboost_plot.png')
    #plot_stump_data(utilities.results_stump_csv, 'decision_stump_plot.png')
    #plot_bagging_data(utilities.results_bagging_csv, 'bagging_plot.png')
    #plot_forest_train_data(utilities.results_forest_csv, 'forest_train_plot.png')
    #plot_forest_test_data(utilities.results_forest_csv, 'forest_test_plot.png')
    #plot_forest_train_test_data(utilities.results_forest_csv, 'forest_plot.png')
    #plot_adaboost_data(utilities.results_adaboost_credit_csv, 'adaboost_credit_plot.png')
    #plot_bagging_data(utilities.results_bagging_credit_csv, 'bagging_credit_plot.png')
    #plot_forest_train_test_data(utilities.results_forest_credit_csv, 'forest_credit_plot.png')

if __name__ == "__main__":
    main()
