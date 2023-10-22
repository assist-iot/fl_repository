import filecmp
import os
import shutil
from http import HTTPStatus

import gridfs
import mongomock.gridfs
from bson import ObjectId
from fastapi.testclient import TestClient
from starlette import status

from application.datamodels.models import MLModel, MLTrainingResults, MLStrategy, \
    MLModelData
from application.main import app

mongomock.gridfs.enable_gridfs_integration()

app.client = mongomock.MongoClient()
client_test = TestClient(app)

# Set up the test environment

# Insert a base model along with its contents placed in a mock GridFS
db_grid = app.client.repository_grid
fs = gridfs.GridFS(db_grid)
with open('../data/temp.zip', 'rb') as f:
    data = f.read()
    model0_id = fs.put(data, filename=f'model/test/0')
model = MLModel(model_name="test", model_version="0", model_id=str(model0_id))
db = app.client.repository
db.models.insert_one(model.dict(by_alias=True))

# Insert some testing results (along with the trained model weights) for the model base
with open('../data/temp.pkl', 'rb') as f:
    data = f.read()
    weight_id = fs.put(data, filename=f'weights/test/0/1')

weights = MLTrainingResults(model_name="test", model_version="0", training_id="1",
                            results={"accuracy": "0.9"},
                            weights_id=str(weight_id))
db = app.client.repository
db.results.insert_one(weights.dict(by_alias=True))

# Insert an additional testing model (without any testing results in place)
with open('../data/temp.zip', 'rb') as f:
    data = f.read()
    model1_id = fs.put(data, filename=f'model/test/1')
model = MLModel(model_name="test", model_version="1", model_id=str(model1_id))
db = app.client.repository
db.models.insert_one(model.dict(by_alias=True))


def test_list_all_base_models_in_database():
    response = client_test.get("/model/")
    assert len(response.json()) == 2
    assert response.status_code == 200


def test_list_all_base_models_that_have_been_trained():
    response = client_test.get("/model/?trained=true")
    assert len(response.json()) == 1
    assert response.json()[0]["model_version"] == "0"
    assert response.status_code == 200


def test_create_base_model():
    new_model = MLModel(model_name="new-model",
                        model_version="new-version").json()
    response = client_test.post("/model", new_model)
    assert response.status_code == status.HTTP_201_CREATED


def test_create_base_model_already_present():
    new_model = MLModel(model_name="test", model_version="0").json()
    response = client_test.post("/model", new_model)
    assert response.status_code == 400


def test_update_model_base_present():
    name_only_meta = "test"
    version_only_meta = "1"
    with open('../data/temp.zip', 'rb') as f:
        files = {"file": f}
        response = client_test.put(f"/model/{name_only_meta}/{version_only_meta}",
                                   files=files)
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    mb_id = db.models.find_one(
        {"model_name": name_only_meta, 'model_version': version_only_meta})
    assert fs.exists(ObjectId(mb_id["model_id"]))


def test_update_model_base_absent():
    name_unavailable = "test"
    version_unavailable = "3"
    with open('../data/temp.zip', 'rb') as f:
        files = {"file": f}
        response = client_test.put(f"/model/{name_unavailable}/{version_unavailable}",
                                   files=files)
    assert response.status_code == 404
    mb_id = db.models.find_one(
        {"model_name": name_unavailable, 'model_version': version_unavailable})
    assert not mb_id


def test_update_model_base_metadata():
    # First check if current meta is different at this point of time
    name_present = "test"
    version_present = "0"
    new_author = "Anonymous"
    current_meta = db.models.find_one(
        {"model_name": name_present, "model_version": version_present})
    assert not current_meta["meta"]
    # Then change it and check again
    new_meta_data = MLModelData(meta={"Author": new_author}).json()
    response = client_test.put(f"/model/meta/{name_present}/{version_present}",
                               new_meta_data)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    new_meta = db.models.find_one(
        {"model_name": name_present, "model_version": version_present})
    assert new_meta["meta"]["Author"] == new_author


def test_update_model_base_metadata_unavailable():
    new_author = "Newbert"
    non_model = "nonexistent"
    non_version = "nonexistent"
    # Then change it and check again
    new_model_data = MLModelData(meta={"Author": new_author}).json()
    response = client_test.put(
        f"/model/meta/{non_model}/{non_version}", new_model_data)
    assert response.status_code == 404
    assert not db.model.find_one({"model_name": non_model, "model_version":
                                  non_version})


def test_download_a_model_base():
    model_name = "test"
    model_version = "0"
    with client_test.get(f"/model/{model_name}/{model_version}", stream=True) as r:
        with open('../data/temp2.zip', 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        assert r.status_code == 200
    assert filecmp.cmp('../data/temp.zip', '../data/temp2.zip', shallow=False)
    os.remove("../data/temp2.zip")


def test_delete_model_base():
    model_name = "test"
    model_version = "0"
    response = client_test.delete(f"/model/{model_name}/{model_version}")
    assert not fs.exists(model0_id)
    assert not db.strategies.find_one(
        {'model_name': model_name, 'model_version': model_version})
    assert response.status_code == HTTPStatus.NO_CONTENT.value


def test_delete_nonexistent_model_base():
    response = client_test.delete(f"/strategy/test/nonexistent")
    assert response.status_code == 404


def test_list_all_training_results_existing_for_a_model_base():
    model_name = "test"
    model_version = "0"
    response = client_test.get(
        f"/training-results/{model_name}/{model_version}")
    assert len(response.json()) == 1
    assert "training_id" in response.json()[0]
    assert response.status_code == 200


def test_list_all_training_results_existing_for_a_model_base_empty():
    model_name = "test"
    model_version = "1"

    response = client_test.get(
        f"/training-results/{model_name}/{model_version}")
    assert len(response.json()) == 0
    assert response.status_code == 200


def test_create_training_results():
    name_present = "test"
    version_present = "0"
    training_id_new = "2"
    new_results = MLTrainingResults(model_name=name_present,
                                    model_version=version_present,
                                    training_id=training_id_new).json()
    response = client_test.post("/training-results", new_results)
    assert response.status_code == status.HTTP_201_CREATED


def test_create_training_results_present():
    name_present = "test"
    version_present = "0"
    training_id_present = "1"
    new_results = MLTrainingResults(model_name=name_present,
                                    model_version=version_present,
                                    training_id=training_id_present).json()
    response = client_test.post("/training-results", new_results)
    assert response.status_code == 400


def test_update_training_results_present():
    name_only_meta = "name_only"
    version_only_meta = "version_only"
    id_only_meta = "id_only"
    results = MLTrainingResults(model_name=name_only_meta,
                                model_version=version_only_meta, training_id=id_only_meta)
    db.results.insert_one(results.dict(by_alias=True))
    with open('../data/temp.pkl', 'rb') as f:
        files = {"file": f}
        response = client_test.put(f"/training-results/{name_only_meta}/"
                                   f"{version_only_meta}/{id_only_meta}",
                                   files=files)
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    tr_id = db.results.find_one(
        {"model_name": name_only_meta, "model_version": version_only_meta,
         "training_id": id_only_meta})
    assert fs.exists(ObjectId(tr_id["weights_id"]))


def test_update_training_results_absent():
    name_unavailable = "nonexistent"
    version_only_meta = "version_only"
    id_only_meta = "id_only"
    assert not db.results.find_one({"model_name": name_unavailable,
                                    "model_version": version_only_meta,
                                    "training_id": id_only_meta})
    with open('../data/temp.pkl', 'rb') as f:
        files = {"file": f}
        response = client_test.put(f"/training-results/{name_unavailable}/"
                                   f"{version_only_meta}/{id_only_meta}",
                                   files=files)
    assert response.status_code == 404
    tr_id = db.results.find_one({"model_name": name_unavailable,
                                 "model_version": version_only_meta,
                                 "training_id": id_only_meta})
    assert not tr_id


def test_download_weights_that_were_the_result_of_a_given_training():
    model_name = "test"
    model_version = "0"
    training_id = "1"
    with client_test.get(f"/training-results/weights/{model_name}/"
                         f"{model_version}/{training_id}", stream=True) as r:
        with open('../data/temp2.pkl', 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        assert r.status_code == 200
    assert filecmp.cmp('../data/temp.pkl', '../data/temp2.pkl', shallow=False)
    os.remove("../data/temp2.pkl")


def test_delete_training_results():
    model_name = "test"
    model_version = "0"
    training_id_present = "1"
    response = client_test.delete(
        f"/training-results/{model_name}/{model_version}/{training_id_present}")
    assert not fs.exists(weight_id)
    assert not db.results.find_one(
        {'model_name': model_name, 'model_version': model_version,
         'training_id': training_id_present})
    assert response.status_code == HTTPStatus.NO_CONTENT.value


def test_delete_nonexistent_model_base():
    response = client_test.delete(f"/training-results/test/0/nonexistent")
    assert response.status_code == 404
