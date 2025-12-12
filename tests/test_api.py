from fastapi.testclient import TestClient

import mlops_rakuten.api as api


class DummyPredictionPipeline:
    """
    Dummy pipeline pour les tests.
    Simule la sortie de prediction_pipeline.run(texts=[...], top_k=...)
    """

    def run(self, texts, top_k=None):
        # Simule une prédiction "top_k" triée, avec proba
        # Retour attendu par l'API : liste (par texte) de listes (categories)
        base = [
            {"prdtypecode": 10, "category_name": "Cat A", "proba": 0.7},
            {"prdtypecode": 20, "category_name": "Cat B", "proba": 0.2},
            {"prdtypecode": 30, "category_name": "Cat C", "proba": 0.1},
        ]
        if top_k is not None:
            base = base[:top_k]
        return [base]  # un seul texte


def test_health_returns_ok(monkeypatch):
    # Monkeypatch du pipeline global dans mlops_rakuten.api
    monkeypatch.setattr(api, "prediction_pipeline", DummyPredictionPipeline())

    client = TestClient(api.app)
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_predictions(monkeypatch):
    monkeypatch.setattr(api, "prediction_pipeline", DummyPredictionPipeline())
    client = TestClient(api.app)

    payload = {
        "designation": "Une designation produit suffisamment longue", "top_k": 2}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["designation"] == payload["designation"]
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 2

    # Vérifier la structure
    first = data["predictions"][0]
    assert set(first.keys()) == {"prdtypecode", "category_name", "proba"}
    assert isinstance(first["prdtypecode"], int)
    assert isinstance(first["category_name"], str)
    assert isinstance(first["proba"], float)


def test_predict_min_length_validation(monkeypatch):
    monkeypatch.setattr(api, "prediction_pipeline", DummyPredictionPipeline())
    client = TestClient(api.app)

    # designation trop courte (<10)
    payload = {"designation": "short", "top_k": 3}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 422  # validation error (Pydantic)


def test_predict_default_top_k(monkeypatch):
    monkeypatch.setattr(api, "prediction_pipeline", DummyPredictionPipeline())
    client = TestClient(api.app)

    # top_k absent => default dans Pydantic (ex: 5)
    payload = {"designation": "Une designation produit suffisamment longue"}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    # Dummy retourne 3 catégories max
    assert len(data["predictions"]) == 3
