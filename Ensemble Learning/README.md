# HW2 - Ensemble Learning

## Required Libraries

To run the code, make sure you have the following libraries installed:

- **Pandas**
- **Numpy**
- **Matplotlib** (Optional, for plotting)
- **Xlrd** (To read xls file for bonus question)

You can install them using:
```bash
pip install pandas numpy matplotlib xlrd
```

## How to Run the Code

Make sure you are inside the `Ensemble Learning` folder.

### Running AdaBoost
To run the AdaBoost implementation:
```bash
python3 adaboost.py
```

### Running Bagging
To run the Bagging implementation:
```bash
python3 bagging.py
```

### Running Random Forest
To run the Random Forest implementation:
```bash
python3 randomforest.py
```

### Running Bias, Variance and Squared Error for Bagging
To run the Random Forest implementation:
```bash
python3 q2-c.py
```

### Running Bias, Variance and Squared Error for Random Forest
To run the Random Forest implementation:
```bash
python3 q2-e.py
```

### Tweaking Parameters
If you want to modify parameters like the **number of iterations** or the **number of trees**, you can adjust the values in the `utilities.py` file located in this folder.

### Generating Plots
To generate plots for the results, run:
```bash
python3 plot_graph.py
```

## Results

The results, including the generated plots and CSV files containing the output from the above code, are stored in the `results` folder.