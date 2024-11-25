import utilities
import PrimalSVM
import DualSVM
import GaussianDualSVM

def main():
    train_features, train_labels = utilities.read_csv(utilities.bank_note_train)
    test_features, test_labels = utilities.read_csv(utilities.bank_note_test)
    PrimalSVM.primal_svm(train_features, train_labels, test_features, test_labels)
    DualSVM.dual_svm(train_features, train_labels, test_features, test_labels)
    GaussianDualSVM.gaussian_dual_svm(train_features, train_labels, test_features, test_labels)
    GaussianDualSVM.overlap(train_features, train_labels)

    return 0

if __name__ == "__main__":
    main()