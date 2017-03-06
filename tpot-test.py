from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import confusion_matrix, classification_report
import dill

pipeline_path = 'nist_pipeline.pkl'  # pipeline dump file


def load_data():
    digits = load_digits()
    return digits.data, digits.target


def train_pipeline():
    exported_pipeline = make_pipeline(
        make_union(VotingClassifier([("est", GradientBoostingClassifier(learning_rate=1.0, max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)),
        make_union(VotingClassifier([("est", MultinomialNB(alpha=44.0, fit_prior=True))]), FunctionTransformer(lambda X: X)),
        KNeighborsClassifier(n_neighbors=3, weights="uniform")
    )

    x, y = load_data()
    exported_pipeline.fit(x, y)

    with open(pipeline_path, 'wb') as f:
        dill.dump(exported_pipeline, f)

    return exported_pipeline


def load_pipeline():
    with open(pipeline_path, 'rb') as f:
        pipeline = dill.load(f)

    return pipeline


def main():
    # pipeline = train_pipeline()
    pipeline = load_pipeline()
    x, y = load_data()
    predicted = pipeline.predict(x)
    print("Classification report:\n{}\n".format(classification_report(y, predicted)))
    print("Confusion matrix:\n{}".format(confusion_matrix(y, predicted)))

main()
