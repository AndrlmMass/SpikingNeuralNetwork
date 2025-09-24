import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from multiple_log_reg.get_data import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    comb_model.fit(comb_train, comb_train_labels)

    # minimal accuracy checks: train / val / test
    def report(name, model, splits):
        X_tr, y_tr, X_v, y_v, X_te, y_te = splits
        acc_tr = accuracy_score(y_tr, model.predict(X_tr))
        acc_v = accuracy_score(y_v, model.predict(X_v))
        acc_te = accuracy_score(y_te, model.predict(X_te))
        print(
            f"{name} accuracy -> train: {acc_tr:.4f}, val: {acc_v:.4f}, test: {acc_te:.4f}"
        )

    report(
        "image",
        image_model,
        (
            image_train,
            image_train_labels,
            image_val,
            image_val_labels,
            image_test,
            image_test_labels,
        ),
    )

    report(
        "audio",
        audio_model,
        (
            audio_train,
            audio_train_labels,
            audio_val,
            audio_val_labels,
            audio_test,
            audio_test_labels,
        ),
    )

    report(
        "combined",
        comb_model,
        (
            comb_train,
            comb_train_labels,
            comb_val,
            comb_val_labels,
            comb_test,
            comb_test_labels,
        ),
    )
