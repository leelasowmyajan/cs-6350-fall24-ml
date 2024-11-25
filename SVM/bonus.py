import utilities
import KernelPerceptron

def main():
    train_features, train_labels = utilities.read_csv(utilities.bank_note_train)
    test_features, test_labels = utilities.read_csv(utilities.bank_note_test)
    KernelPerceptron.kernel_perceptron(train_features, train_labels, test_features, test_labels)
    return 0

if __name__ == "__main__":
    main()