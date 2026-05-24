from fastapi.testclient import TestClient

from app.main import app


def test_root_redirects_to_upload():
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].endswith("/static/index.html")


def test_static_files_served():
    client = TestClient(app)
    for path in ("/static/index.html", "/static/people.html",
                 "/static/processing.html", "/static/done.html",
                 "/static/js/api.js", "/static/js/progress.js"):
        r = client.get(path)
        assert r.status_code == 200, path
