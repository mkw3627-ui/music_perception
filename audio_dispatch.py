"""Source candidates and silent, replay-safe AI delivery for N.E.K.O.

No SDK imports, audio capture, network or other plugin access here.
Sources are process-scoped, *not* browser-tab-scoped. External visual
judgements are labelled as claims rather than proven acoustic ground truth.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict

TAGS = frozenset({'music', 'video', 'other', 'unknown'})
EVIDENCE = frozenset({'ai_window_review', 'window_plugin', 'user_review', 'title_process'})


def source_id(app: dict) -> str:
    """A PID is never sufficient without its creation token."""
    return 'app:{}:{}'.format(int(app['pid']), str(app['created']))


class SourceLabels:
    """Bounded, short-lived external revisions to initial process-name hints.

    The audio plugin never claims to have seen a video frame. The other plugin
    may report a window-derived label via a permissioned entry point.
    """
    def __init__(self, *, ttl_s=600, limit=128, clock=None):
        self._clock = clock or time.monotonic
        self.ttl_s = ttl_s
        self.limit = limit
        self._lock = threading.RLock()
        self._data = OrderedDict()

    def update(self, identity: str, tag: str, *, evidence: str, current_sources,
               ttl_s: float | None = None):
        if tag not in TAGS or evidence not in EVIDENCE:
            raise ValueError('invalid_label_or_evidence')
        if not isinstance(identity, str) or len(identity)>120:
            raise ValueError('invalid_source_id')
        live = next((r for r in current_sources if source_id(r) == identity), None)
        if live is None or not live.get('audible'):
            raise ValueError('source_not_currently_audible')
        if live.get('host_ai'):
            raise ValueError('fixed_ai_host_cannot_be_relabelled')
        now=self._clock()
        # A reviewer may ask for a longer-lived label than the default TTL;
        # an invalid or non-positive value falls back instead of disabling expiry.
        life = (float(ttl_s) if isinstance(ttl_s,(int,float)) and not isinstance(ttl_s,bool)
                and ttl_s > 0 else self.ttl_s)
        with self._lock:
            previous = self._data.get(identity)
            if tag == 'unknown':
                was_present = identity in self._data
                self._data.pop(identity, None)
                return {'changed':was_present, 'tag':'unknown', 'state':'candidate'}
            if previous and previous['tag']==tag and previous['evidence']==evidence:
                previous['expires_at']=now+life
                return {'changed':False,'tag':tag,'state':'external_review'}
            self._data[identity]={'tag':tag,'evidence':evidence,
                                  'expires_at':now+life}
            self._data.move_to_end(identity)
            while len(self._data)>self.limit:
                self._data.popitem(last=False)
            return {'changed':True,'tag':tag,'state':'external_review'}

    def apply(self, app: dict) -> dict:
        result = dict(app)
        if result.get('host_ai'):
            result.update(tag='ai',label='AI 主程序 · 不采集',confirmed=True,
                          tag_status='fixed',tag_origin='host_process')
            return result
        identity = source_id(result)
        now=self._clock()
        with self._lock:
            review=self._data.get(identity)
            if review is not None and review['expires_at'] < now:
                self._data.pop(identity,None)
                review=None
        if review:
            tag=review['tag']
            result.update(tag=tag,
                          label={'music':'音乐 · 外部窗口复核','video':'视频 · 外部窗口复核',
                                 'other':'其他声音 · 外部窗口复核'}[tag],
                          confirmed=False,tag_status='external_review',
                          tag_origin=review['evidence'],
                          candidate_tag=result.get('tag','unknown'))
        else:
            result.update(tag_status='candidate',tag_origin='local_hint',confirmed=False)
        return result

    def peek(self, identity: str) -> dict:
        """Read a review record without requiring a fresh audible scan.

        A reviewed source must stay resolvable while that app is momentarily
        quiet or briefly dropped by a scan; otherwise the permission silently
        reverts to 'candidate' and the video/dialogue channel stops.
        """
        if not isinstance(identity, str) or not identity:
            return {}
        now = self._clock()
        with self._lock:
            review = self._data.get(identity)
            if review is None:
                return {}
            if review['expires_at'] < now:
                self._data.pop(identity, None)
                return {}
            tag = review['tag']
            return {'tag': tag, 'tag_status': 'external_review',
                    'tag_origin': review['evidence'],
                    'label': {'music': '音乐 · 外部窗口复核',
                              'video': '视频 · 外部窗口复核',
                              'other': '其他声音 · 外部窗口复核'}.get(tag, '外部复核')}

    def drop_disappeared(self, live_ids):
        """Expire only aged identities, not short quiet pauses.

        A source is *never* exposed as currently audible unless a fresh Windows
        peak reading exists. An old review can be reused for the same PID and
        creation token after a temporary silence; not for a new process.
        """
        now=self._clock()
        with self._lock:
            for identity,item in list(self._data.items()):
                if item['expires_at'] < now:
                    self._data.pop(identity,None)


class PrivateAudioBus:
    """Exactly-once-for-identical-snapshot within a plugin run, best effort.

    A rejected/failed delivery is retryable. A newer state does not get
    suppressed just because the same coalescing channel was used previously.
    This cannot prove that the LLM consumed the private observation.
    """
    def __init__(self, push):
        self._push=push
        self._lock=threading.RLock()
        self._delivered={}
        self._pending={}
        self.last_rejection=None

    def publish(self, topic: str, payload: dict, *, signature=None,
                coalesce_key=None, max_bytes=48000):
        if not isinstance(payload,dict) or not topic.replace('_','').isalnum():
            raise ValueError('invalid_private_audio_packet')
        item={**payload, 'privacy':'ai_read_only',
              'display_in_chat':False,'requests_response':False,
              'assistant_decides_whether_and_how_to_refer_to_data':True}
        text='[内部音频感知数据 / 非用户消息 / 不构成发言请求]\n'+json.dumps(
            item,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        if len(text.encode('utf-8'))>max_bytes:
            raise ValueError('private_audio_payload_too_large')
        digest=hashlib.sha256((signature if isinstance(signature,str) else text).encode('utf-8')).hexdigest()
        with self._lock:
            if self._delivered.get(topic)==digest or self._pending.get(topic)==digest:
                self.last_rejection=None  # deduplicated is not a delivery error
                return False
            self._pending[topic]=digest
        success=False
        try:
            receipt=self._push(source='music_perception',visibility=[],ai_behavior='read',
                               coalesce_key=coalesce_key or 'music_perception_private_'+topic,
                               parts=[{'type':'text','text':text}], priority=1)
            if isinstance(receipt,dict):
                success=receipt.get('submitted',False) is True
                self.last_rejection=None if success else str(receipt.get('reason','not_accepted'))[:120]
            else:
                success=getattr(receipt,'submitted',True) is not False
                self.last_rejection=None if success else 'not_accepted'
            return success
        finally:
            with self._lock:
                if self._pending.get(topic)==digest:
                    self._pending.pop(topic,None)
                if success:
                    self._delivered[topic]=digest
