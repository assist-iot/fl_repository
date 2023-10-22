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
    MLStrategyData
from application.main import app

mongomock.gridfs.enable_gridfs_integration()

app.client = mongomock.MongoClient()
client_test = TestClient(app)

# Set up the test environment

# Insert a strategy

test_strategy_name = "test-strategy"
test_strategy_description = "A test strategy"

db_grid = app.client.repository_grid
fs = gridfs.GridFS(db_grid)
with open('../data/temp.pkl', 'rb') as f:
    data = f.read()
    strategy_id = fs.put(data, filename=f'strategy/{test_strategy_name}')
model = MLStrategy(strategy_name=test_strategy_name,
                   strategy_description="A test strategy",
                   strategy_id=str(strategy_id))
db = app.client.repository
db.strategies.insert_one(model.dict(by_alias=True))


# Begin the tests

def test_list_all_strategies_in_database():
    response = client_test.get("/strategy")
    assert len(response.json()) == 1
    assert response.status_code == 200


def test_create_strategy():
    new_strategy = MLStrategy(strategy_name="new-strategy")
    response = client_test.post("/strategy", new_strategy)
    assert response.status_code == status.HTTP_201_CREATED


def test_create_strategy_present():
    new_strategy = MLStrategy(strategy_name=test_strategy_name).json()
    response = client_test.post("/strategy", new_strategy)
    assert response.status_code == 400


def test_update_strategy_base_present():
    name_only_meta = "name_only"
    strat = MLStrategy(strategy_name=name_only_meta)
    db.strategies.insert_one(strat.dict(by_alias=True))
    with open('../data/temp.pkl', 'rb') as f:
        files = {"file": f}
        response = client_test.put(f"/strategy/{name_only_meta}",
                                   files=files)
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    sb_id = db.strategies.find_one(
        {"strategy_name": name_only_meta})
    assert fs.exists(ObjectId(sb_id["strategy_id"]))


def test_update_strategy_base_absent():
    name_unavailable = "nonexistent"
    assert not db.strategies.find_one({"strategy_name": name_unavailable})
    with open('../data/temp.pkl', 'rb') as f:
        files = {"file": f}
        response = client_test.put(f"/strategy/{name_unavailable}",
                                   files=files)
    assert response.status_code == 404
    mb_id = db.strategies.find_one({"strategy_name": name_unavailable})
    assert not mb_id


def test_update_strategy_metadata():
    # First check if current meta is different at this point of time
    new_author = "Anonymous"
    current_meta = db.strategies.find_one(
        {"strategy_name": test_strategy_name})
    assert not current_meta["meta"]
    # Then change it and check again
    new_strategy_data = MLStrategyData(meta={"Author": new_author}).json()
    response = client_test.put(
        f"/strategy/meta/{test_strategy_name}", new_strategy_data)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    new_meta = db.strategies.find_one({"strategy_name": test_strategy_name})
    assert new_meta["meta"]["Author"] == new_author


def test_update_strategy_metadata_unavailable():
    new_author = "Newbert"
    # Then change it and check again
    new_strategy_data = MLStrategyData(meta={"author": new_author}).json()
    response = client_test.put(
        f"/strategy/meta/nonexistent", new_strategy_data)
    assert response.status_code == 404
    assert not db.strategies.find_one({"strategy_name": "nonexistent"})


def test_download_a_strategy_base():
    with client_test.get(f"/strategy/{test_strategy_name}", stream=True) as r:
        with open('../data/temp2.pkl', 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        assert r.status_code == 200
    assert filecmp.cmp('../data/temp.pkl', '../data/temp2.pkl', shallow=False)
    os.remove("../data/temp2.pkl")


def test_delete_strategy():
    response = client_test.delete(f"/strategy/{test_strategy_name}")
    assert not fs.exists(strategy_id)
    assert not db.strategies.find_one({'strategy_name': test_strategy_name})
    assert response.status_code == HTTPStatus.NO_CONTENT.value


def test_delete_nonexistent_strategy():
    response = client_test.delete(f"/strategy/nonexistent")
    assert response.status_code == 404
