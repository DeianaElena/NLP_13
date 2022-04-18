# %%
# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []

    # TODO: determine the majority class based on the training data
    # ...
    majority_class = "X"
    predictions = []
    for instance in testinput:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance, instance_predictions)

    # TODO: calculate accuracy for the test input
    # ...
    return accuracy, predictions





if __name__ == '__main__':
    train_path = "./data/preprocessed/train"
    val_path = "./data/preprocessed/val"
    test_path = "./data/preprocessed/test"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "/sentences.txt", encoding="utf8") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "/labels.txt", encoding="utf8") as label_file:
        train_labels = label_file.readlines()


    with open(val_path + "/sentences.txt", encoding="utf8") as val_file:
        val_sentences = val_file.readlines()

    with open(train_path + "/labels.txt", encoding="utf8") as val_label_file:
        val_labels = val_label_file.readlines()
    with open(test_path + "/sentences.txt") as testfile:
        testinput = testfile.readlines()

    with open(test_path + "/labels.txt", encoding="utf8") as test_labelfile:
        testlabels = test_labelfile.readlines()
    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, testinput)

    # TODO: output the predictions in a suitable way so that you can evaluate them