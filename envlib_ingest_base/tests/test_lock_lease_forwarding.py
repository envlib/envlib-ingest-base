"""lock_lease reaches every write session the toolkit opens, and nothing changes when it is
not given (fakes and older callers never see the keyword)."""
import contextlib

import pytest

from envlib_ingest_base import tsortho
from envlib_ingest_base.tests.test_telemetry_layout import ENC, STNS3, _series3, make_meta, stations_dict


class RecordingCat:
    def __init__(self):
        self.calls = []

    def publish(self, path, member_conn, rcg_conn, **kw):
        self.calls.append(kw)
        return 'ok'


@pytest.mark.parametrize('lease', [None, 1500])
def test_update_and_publish_forwards_the_lease_to_both_sessions(monkeypatch, lease):
    opens = []

    @contextlib.contextmanager
    def fake_open(conn, path, flag, **kw):
        opens.append((flag, kw))
        yield object()
    monkeypatch.setattr(tsortho, 'open_edataset', fake_open)
    monkeypatch.setattr(tsortho, 'merge_dataset', lambda ds, stations, series, variable: {'merged': True})
    cat = RecordingCat()
    kw = {} if lease is None else {'lock_lease': lease}
    out = tsortho.update_and_publish(cat, 'w.cfdb', 'member', 'rcg', {}, {}, variable='v', **kw)
    assert out['merge'] == {'merged': True}
    assert opens == [('w', kw)]                     # the merge session
    assert cat.calls == [kw]                        # the publish session


class FakeLock:
    def __init__(self, holds, reason=None):
        self.holds, self.lost_reason, self.verified = holds, reason, 0

    def verify(self):
        self.verified += 1
        return self.holds


class FakeEDataset:
    def __init__(self, lock):
        self._blt = type('Blt', (), {'lock': lock})()


def _run_with_lock(monkeypatch, tmp_path, lock):
    @contextlib.contextmanager
    def fake_open(conn, path, flag, **kw):
        yield FakeEDataset(lock)
    monkeypatch.setattr(tsortho, 'open_edataset', fake_open)
    monkeypatch.setattr(tsortho, 'merge_dataset', lambda ds, stations, series, variable: {'merged': True})
    work = tmp_path / 'w.cfdb'
    for name in ('w.cfdb', 'w.cfdb.remote_index', 'w.cfdb.changelog', 'w.cfdbx', 'other.cfdb'):
        (tmp_path / name).write_bytes(b'x')
    cat = RecordingCat()
    return cat, work, lambda: tsortho.update_and_publish(cat, work, 'm', 'r', {}, {}, variable='v', lock_lease=1500)


@pytest.mark.parametrize('reason', ['lease_expired', 'broken'])
def test_a_merge_that_lost_its_lock_is_discarded_and_not_published(monkeypatch, tmp_path, reason):
    """Round ecan-lease-code-1, finding 1: a stalled merge must not be published over a newer commit."""
    lock = FakeLock(holds=False, reason=reason)
    cat, work, run = _run_with_lock(monkeypatch, tmp_path, lock)
    with pytest.raises(tsortho.LockLostError, match='merge session lost its write lock'):
        run()
    assert lock.verified == 1
    assert cat.calls == [], 'nothing may be published'
    # the working cache and its sidecars are gone; files that merely share a prefix are kept
    assert sorted(p.name for p in tmp_path.iterdir()) == ['other.cfdb', 'w.cfdbx']


def test_a_merge_whose_lock_holds_is_published(monkeypatch, tmp_path):
    lock = FakeLock(holds=True)
    cat, work, run = _run_with_lock(monkeypatch, tmp_path, lock)
    assert run()['publish'] == 'ok'
    assert lock.verified == 1
    assert cat.calls == [{'lock_lease': 1500}]
    assert work.exists()


@pytest.mark.parametrize('lease', [None, 1500])
def test_build_and_publish_forwards_the_lease(tmp_path, lease):
    cat = RecordingCat()
    kw = {} if lease is None else {'lock_lease': lease}
    tsortho.build_and_publish(cat, tmp_path / 'p.cfdb', None, None, make_meta(), stations_dict(STNS3),
                              _series3(60, with_qc=False), group_bytes=1 << 20, **kw, **ENC)
    assert cat.calls == [dict(group_bytes=1 << 20, **kw)]
