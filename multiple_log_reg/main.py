import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from linear_classifier.get_data import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    # get data (audio, image, and combined)
    (
        image_train,
        image_train_labels,
        image_val,
        image_val_labels,
        image_test,
        image_test_labels,
        audio_train,
        audio_train_labels,
        audio_val,
        audio_val_labels,
        audio_test,
        audio_test_labels,
        comb_train,
        comb_train_labels,
        comb_val,
        comb_val_labels,
        comb_test,
        comb_test_labels,
    ) = get_data(imageMNIST=True, audioMNIST=True, combined=True)

    # Create a list to store results
    results = {"image": [], "audio": [], "combined": []}

    # loop over training and testing data to gauge performance
    for _ in tqdm(range(1)):
        # train models
        audio_model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000
        )
        image_model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000
        )
        comb_model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000
        )

        audio_model.fit(audio_train, audio_train_labels)
        image_model.fit(image_train, image_train_labels)
        if comb_train is not None:
            comb_model.fit(comb_train, comb_train_labels)

        # evaluate
        y_pred_audio = audio_model.predict(audio_test)
        results["audio"].append(accuracy_score(audio_test_labels, y_pred_audio))

        y_pred_image = image_model.predict(image_test)
        results["image"].append(accuracy_score(image_test_labels, y_pred_image))

        if comb_test is not None:
            y_pred_comb = comb_model.predict(comb_test)
            results["combined"].append(accuracy_score(comb_test_labels, y_pred_comb))

    # print average results, variance and std for each
    for key, value in results.items():
        print(f"{key}: {np.mean(value)}, {np.var(value)}, {np.std(value)}")

    # plot results as a box plot
    plt.boxplot(results.values())
    plt.xticks(range(1, len(results) + 1), results.keys())
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Linear Classifiers")
    plt.savefig("linear_classifier_results.png")
