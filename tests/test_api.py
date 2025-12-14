import requests
import pytest

# Configuration commune
BASE_URL = "http://localhost:8000"
AUTH = ("admin", "4dm1N")
USER = ("alice", "wonderland") 

# Liste des utilisateurs à tester
USERS = [
    ("alice", "wonderland"),
    ("bob", "builder"),
]

# Fixture pour éviter la répétition de l'URL de base
@pytest.fixture
def base_url():
    return BASE_URL

# Fixture pour l'authentification
@pytest.fixture
def auth():
    return AUTH

# Fixture pour l'authentification user 
@pytest.fixture
def user():
    return USER

# 1. Test de la page de bienvenue
def test_welcome_page(base_url):
    response = requests.get(f"{base_url}/", headers={"accept": "application/json"})
    assert response.status_code == 200
    # Ajoutez ici les assertions spécifiques à la réponse attendue
    # Exemple : assert "welcome" in response.json()

# 2. Test de l'endpoint de santé
def test_health_endpoint(base_url):
    response = requests.get(f"{base_url}/health", headers={"accept": "application/json"})
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "healthy"  # Adaptez selon la réponse attendue

# 3.Test de l'endpoint load-and-train (POST avec upload de fichiers)
def test_load_and_train_endpoint(base_url, auth):
    files = {
        "x_train_file": open("data/raw/X_train_update.csv", "rb"),
        "y_train_file": open("data/raw/Y_train_CVw08PX.csv", "rb")
    }
    response = requests.post(
        f"{base_url}/admin/load-and-train",
        auth=auth,
        files=files
    )
    assert response.status_code == 200  # ou le code attendu
    # Ajoutez ici les assertions spécifiques à la réponse
    # Exemple : assert "success" in response.json()

# 6. test des prédictions sur un texte
@pytest.mark.parametrize("user", USERS)
def test_predict_endpoint(base_url, user):
    data = {
        "texts": ["Porte Flamme Gaxix - Flamebringer Gaxix - 136/220 - U - Twilight Of The Dragons" ]
    }
    response = requests.post(
        f"{base_url}/predict",
        auth=user,
        json=data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert "predictions" in response_data
    assert len(response_data["predictions"]) == 1



