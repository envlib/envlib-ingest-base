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


@pytest.mark.parametrize('lease', [None, 1500])
def test_build_and_publish_forwards_the_lease(tmp_path, lease):
    cat = RecordingCat()
    kw = {} if lease is None else {'lock_lease': lease}
    tsortho.build_and_publish(cat, tmp_path / 'p.cfdb', None, None, make_meta(), stations_dict(STNS3),
                              _series3(60, with_qc=False), group_bytes=1 << 20, **kw, **ENC)
    assert cat.calls == [dict(group_bytes=1 << 20, **kw)]
