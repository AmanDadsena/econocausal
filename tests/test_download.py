"""Tests for the download step, using a fake server (no network access)."""

import io
import zipfile

import pytest

from econocausal.data import download


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BrokenResponse(FakeResponse):
    def read(self, *args):
        raise ConnectionError("connection dropped")


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    return tmp_path


def serve(monkeypatch, payload: bytes, response_cls=FakeResponse):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return response_cls(payload)

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_downloads_and_extracts_the_excel_file(raw_dir, monkeypatch):
    calls = serve(monkeypatch, make_zip({"online_retail_II.xlsx": b"excel bytes", "readme.txt": b"x"}))
    path = download.download_dataset()
    assert path == raw_dir / download.EXCEL_NAME
    assert path.read_bytes() == b"excel bytes"
    assert calls == [download.DATASET_URL]
    assert sorted(p.name for p in raw_dir.iterdir()) == [download.EXCEL_NAME]  # zip removed


def test_skips_the_download_when_the_file_exists(raw_dir, monkeypatch):
    (raw_dir / download.EXCEL_NAME).write_bytes(b"already here")
    calls = serve(monkeypatch, b"")
    download.download_dataset()
    assert calls == []


def test_force_downloads_again(raw_dir, monkeypatch):
    (raw_dir / download.EXCEL_NAME).write_bytes(b"old")
    serve(monkeypatch, make_zip({"data/Online_Retail_II.XLSX": b"new"}))
    assert download.download_dataset(force=True).read_bytes() == b"new"


def test_rejects_a_file_that_is_not_a_zip(raw_dir, monkeypatch):
    serve(monkeypatch, b"<html>error page</html>")
    with pytest.raises(ValueError, match="not a valid zip"):
        download.download_dataset()
    assert not (raw_dir / download.EXCEL_NAME).exists()


def test_rejects_a_zip_without_an_excel_file(raw_dir, monkeypatch):
    serve(monkeypatch, make_zip({"readme.txt": b"no data"}))
    with pytest.raises(ValueError, match="No .xlsx file"):
        download.download_dataset()


def test_failed_download_leaves_no_partial_files(raw_dir, monkeypatch):
    serve(monkeypatch, b"", response_cls=BrokenResponse)
    with pytest.raises(ConnectionError):
        download.download_dataset()
    assert list(raw_dir.iterdir()) == []


def test_failed_extraction_does_not_leave_a_file_that_looks_finished(raw_dir, monkeypatch):
    serve(monkeypatch, make_zip({"online_retail_II.xlsx": b"x" * 1000}))

    def broken_copy(src, dst):
        dst.write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(download.shutil, "copyfileobj",
                        lambda src, dst: broken_copy(src, dst) if dst.name.endswith(".xlsx.part") else dst.write(src.read()))
    with pytest.raises(OSError, match="disk full"):
        download.download_dataset()
    assert not (raw_dir / download.EXCEL_NAME).exists()
    assert not any(p.suffix == ".part" for p in raw_dir.iterdir())
