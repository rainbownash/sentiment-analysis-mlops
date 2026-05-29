from src.models.train import train_model
from src.models.evaluation import get_metrics

def run_experiment(model,
                   X_train,
                   X_test,
                   y_train,
                   y_test,
                   representation,
                   classifier_name):

    model = train_model(model, X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = get_metrics(
        y_test,
        y_pred,
        representation,
        classifier_name
    )

    return model, metrics, y_pred