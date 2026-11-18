"""Capture startup error must never leave background audio/lyrics workers."""
import asyncio

from music_perception.auto_listener import ListenerError


def test_failure_after_worker_start_rolls_back_all_owned_resources(plugin_fixture,monkeypatch,tmp_path):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        model=tmp_path/'yamnet.tflite'
        model.write_bytes(b'fake test model')
        monkeypatch.setattr(plugin,'_yamnet_model',lambda:model)
        monkeypatch.setattr('music_perception.auto_listener.YamnetMusicGate',
                            lambda *args,**kwargs:object())
        cleanup=[]
        monkeypatch.setattr(plugin,'_start_background_processor',lambda *args:cleanup.append('workers_started'))
        monkeypatch.setattr(plugin,'_stop_background_processor',lambda:cleanup.append('workers_stopped'))
        import music_perception
        class BrokenListener:
            def __init__(self,**kw):
                self.running=False
            def start(self):
                raise ListenerError('synthetic_wasapi_start_failure')
            def stop(self):
                cleanup.append('listener_stopped')
                return True
        monkeypatch.setattr(music_perception,'LiveListener',BrokenListener)
        result=await plugin._start_live()
        assert not result['ok'] and 'synthetic_wasapi' in result['error']
        assert cleanup==['workers_started','listener_stopped','workers_stopped']
        assert plugin._listener is None
        assert plugin._source_arbiter_live is False
        await plugin.shutdown()
    asyncio.run(scenario())
