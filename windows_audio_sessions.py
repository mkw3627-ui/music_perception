"""Read-only Windows Core Audio *render session* peak meters.

The scanner does not open a loopback recording stream and never reads PCM.
It enumerates active render endpoints and their session meters to find audio
sessions whose *current* peak is above the user-visible threshold. A reported
PID can represent a shared session (or a browser helper), so this is evidence
of a sound-producing session, not proof of unique app-to-audio ownership.

Uses only stdlib ctypes; native COM calls are made only on Windows, within
an executor thread. Refuses to silently list all running processes on errors.
"""
from __future__ import annotations
import math
import sys
import time
from uuid import UUID

DEFAULT_MIN_PEAK = 0.004  # about -48 dBFS; peaks below this are not confidently audible


def level_dbfs(peak: float) -> float:
    if not math.isfinite(peak) or peak <= 0:
        return -120.0
    return round(20 * math.log10(max(1e-6, min(1.0, peak))), 1)


def active_session_rows(readings, *, min_peak=DEFAULT_MIN_PEAK):
    """Merge repeated meter samples/endpoints; retain only audible single-PID sessions.

    This pure function is separately regression-tested without Windows.
    """
    output = {}
    for sample in readings:
        try:
            pid = int(sample['pid'])
            peak = float(sample.get('peak', 0))
            if (pid <= 4 or not math.isfinite(peak) or not 0 <= peak <= 1 or
                    peak < min_peak or sample.get('shared_pid') or sample.get('system_audio')):
                continue
            row = output.setdefault(pid, {'pid':pid, 'peak':0.0, 'sample_count':0})
            row['peak'] = max(row['peak'], peak)
            row['sample_count'] += 1
        except (ValueError, TypeError, KeyError):
            continue
    return [{**r, 'level_dbfs':level_dbfs(r['peak'])}
            for r in sorted(output.values(), key=lambda x:-x['peak'])]


def _com_peak_readings():
    """Take a single snapshot of session peaks on all active render devices.

    Uses the documented Core Audio vtable order. Every COM interface is
    released on the same COM-initialized worker thread, including error paths.
    """
    if sys.platform != 'win32':
        raise OSError('Windows Core Audio 音量会话扫描仅支持 Windows')
    import ctypes as c
    from ctypes import wintypes as w
    class GUID(c.Structure):
        _fields_ = [('Data1',w.DWORD),('Data2',w.WORD),('Data3',w.WORD),('Data4',c.c_ubyte*8)]
    def guid(value):
        return GUID.from_buffer_copy(UUID(value).bytes_le)
    def vcall(obj, slot, *argtypes):
        table = c.cast(obj, c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(c.c_long,c.c_void_p,*argtypes)(table[slot])
    def checked(value, label):
        if value < 0:
            raise OSError(f'{label}: HRESULT 0x{(value & 0xffffffff):08x}')
    def release(ptr):
        if ptr and ptr.value:
            vcall(ptr,2)(ptr)
            ptr.value = None
    def query(obj, ident):
        result = c.c_void_p()
        checked(vcall(obj,0,c.POINTER(GUID),c.POINTER(c.c_void_p))
                (obj,c.byref(ident),c.byref(result)),'QueryInterface')
        return result
    CLSID = guid('bcde0395-e52f-467c-8e3d-c4579291692e')
    IID_ENUM = guid('a95664d2-9614-4f35-a746-de8db63617e6')
    IID_MANAGER = guid('77aa99a0-1bd6-484f-8bc7-2c654c9a9b6f')
    IID_CTRL2 = guid('bfb7ff88-7239-4fc9-8fa2-07c950be9c6d')
    IID_METER = guid('c02216f6-8c67-4b5b-9d00-d008e73e0064')
    ole = c.OleDLL('ole32')
    ole.CoInitializeEx.argtypes = [c.c_void_p,w.DWORD]
    ole.CoInitializeEx.restype = c.c_long
    initialized = ole.CoInitializeEx(None,0)  # MTA, never execute on UI thread
    if initialized < 0 and (initialized & 0xffffffff) != 0x80010106:
        checked(initialized,'CoInitializeEx')
    uninitialize = initialized in (0,1)
    ole.CoCreateInstance.argtypes = [c.POINTER(GUID),c.c_void_p,w.DWORD,
                                     c.POINTER(GUID),c.POINTER(c.c_void_p)]
    ole.CoCreateInstance.restype = c.c_long
    enum = c.c_void_p()
    results = []
    try:
        checked(ole.CoCreateInstance(c.byref(CLSID),None,23,c.byref(IID_ENUM),c.byref(enum)),
                'CoCreateInstance')
        coll = c.c_void_p()
        try:
            # EDataFlow.eRender=0, DEVICE_STATE_ACTIVE=1
            checked(vcall(enum,3,c.c_int,w.DWORD,c.POINTER(c.c_void_p))
                    (enum,0,1,c.byref(coll)),'EnumAudioEndpoints')
            total = w.UINT()
            checked(vcall(coll,3,c.POINTER(w.UINT))(coll,c.byref(total)),
                    'IMMDeviceCollection.GetCount')
            for di in range(min(total.value,24)):
                dev = c.c_void_p()
                try:
                    checked(vcall(coll,4,w.UINT,c.POINTER(c.c_void_p))
                            (coll,di,c.byref(dev)),'IMMDeviceCollection.Item')
                    mgr = c.c_void_p()
                    try:
                        checked(vcall(dev,3,c.POINTER(GUID),w.DWORD,c.c_void_p,c.POINTER(c.c_void_p))
                                (dev,c.byref(IID_MANAGER),23,None,c.byref(mgr)),
                                'IMMDevice.Activate')
                        sess_enum = c.c_void_p()
                        try:
                            # IAudioSessionManager2 = IAudioSessionManager + GetSessionEnumerator (slot 5)
                            checked(vcall(mgr,5,c.POINTER(c.c_void_p))
                                    (mgr,c.byref(sess_enum)),'GetSessionEnumerator')
                            count = c.c_int()
                            checked(vcall(sess_enum,3,c.POINTER(c.c_int))
                                    (sess_enum,c.byref(count)),'GetCount')
                            for si in range(min(count.value,512)):
                                control = c.c_void_p()
                                try:
                                    checked(vcall(sess_enum,4,c.c_int,c.POINTER(c.c_void_p))
                                            (sess_enum,si,c.byref(control)),'GetSession')
                                    state=c.c_int()
                                    if vcall(control,3,c.POINTER(c.c_int))(control,c.byref(state)) != 0 or state.value!=1:
                                        continue
                                    ctl2 = c.c_void_p()
                                    meter = c.c_void_p()
                                    try:
                                        ctl2 = query(control,IID_CTRL2)
                                        pid = w.DWORD()
                                        hr = vcall(ctl2,14,c.POINTER(w.DWORD))(ctl2,c.byref(pid))
                                        # AUDCLNT_S_NO_SINGLE_PROCESS -> a shared session.
                                        if hr != 0 or pid.value <= 4:
                                            continue
                                        meter = query(control,IID_METER)
                                        peak=c.c_float()
                                        checked(vcall(meter,3,c.POINTER(c.c_float))
                                                (meter,c.byref(peak)),'GetPeakValue')
                                        results.append({'pid':int(pid.value),'peak':float(peak.value)})
                                    except OSError:
                                        # Some Windows render sessions don't expose all interfaces.
                                        continue
                                    finally:
                                        release(meter);release(ctl2)
                                finally:
                                    release(control)
                        finally:
                            release(sess_enum)
                    finally:
                        release(mgr)
                finally:
                    release(dev)
        finally:
            release(coll)
    finally:
        release(enum)
        if uninitialize:
            ole.CoUninitialize()
    return results


def scan_audible_sessions(*, min_peak=DEFAULT_MIN_PEAK, sample_interval=0.12):
    """Use two brief peak reads; session may be active but momentarily quiet.

    A call is user-triggered, so no continuous background recording is started.
    """
    first = _com_peak_readings()
    if sample_interval > 0:
        time.sleep(min(.35,max(0.0,sample_interval)))
    return active_session_rows(first + _com_peak_readings(),min_peak=min_peak)
