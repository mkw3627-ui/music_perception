import {
  Page, Card, Stack, Text, StatusBadge, StatCard, Grid, Alert, Modal,
  Button, Switch, SegmentedControl, useState, useEffect,
} from "@neko/plugin-ui"
import type { PluginSurfaceProps } from "@neko/plugin-ui"
import { Onboarding, qualities, singingOptions, stemChoices } from "./onboarding"

const UI_PROTOCOL_BUILD = "0.9.24-commentary-r1"
// N.E.K.O Hosted TSX; the same native UI kit used by 为她所控.
// No external browser, CDN, React build or extra local web server.
// Supersedes "0.9.0-audible-app-tags-r1". Backend compatibility interface
// "set_ai_voice_playback" remains registered for host TTS callbacks ONLY;
// there is no redundant user-facing AI-voice toggle in this unified console.
// Separate music/singing checkpoint choices. The original `qualities` array
// remains the VIDEO SPEECH model catalog, preserving downloaded Whisper files.
const tabs = [
  {value:"listen", label:"音频"},
  {value:"source", label:"应用声道"},
  {value:"video", label:"视频"},
  {value:"model", label:"模型"},
  {value:"diagnose", label:"检测"},
  {value:"groove", label:"律动"},
  {value:"settings", label:"设置"},
]
const devices = [{value:"cpu", label:"系统内存 · CPU"}, {value:"cuda", label:"显存 · NVIDIA GPU"}]
const languages = [
  {value:"auto", label:"自动"}, {value:"zh", label:"中文"},
  {value:"ja", label:"日语"}, {value:"en", label:"英语"},
]
const sources = [{value:"mirror", label:"国内镜像优先"}, {value:"official", label:"官方优先"}]
const tierLabels: Record<string,string> = {low:"低", medium:"中", high:"高", ultra:"极高"}
const enhanceModes=[{value:"off",label:"关闭"},{value:"gentle",label:"轻柔"},
 {value:"balanced",label:"标准"},{value:"strong",label:"强化"}]
const modelStates: Record<string,string> = {
  verified:"已校验", checking:"检查中", missing:"未下载", incomplete:"未完成",
  present_unverified:"待校验", downloading:"下载中", verifying:"校验中",
  failed:"下载失败", interrupted:"已中断", unreadable:"文件不可读",
  running:"准备中", ready:"就绪",
}
const lyricStates: Record<string,string> = {
  collecting_audio:"采集中", queued:"排队中", preparing_model:"准备模型",
  transcribing:"识别中", asr_hypothesis:"歌词已更新", waiting_for_vocals:"等待歌声",
  no_reliable_lyrics:"等待清晰歌声", model_download_waiting_network:"权重未就绪",
  plugin_paused_gpu_released:"已暂停", unavailable:"故障",
}
function unpack(response:any):any {
  let data = response
  for (let i=0; i<6; i++) {
    if (!data || typeof data !== "object") break
    if (data.success === false) throw Error(String(data.error || "操作失败"))
    // Hosted bridge / SDK Ok(data) may add one or more wrappers.
    if (data.value !== undefined) {data=data.value;continue}
    if (data.data !== undefined && data.ok === undefined) {data=data.data;continue}
    if (data.result !== undefined && data.ok === undefined) {data=data.result;continue}
    break
  }
  if (data && data.ok === false) throw Error(String(data.error || data.message || "操作失败"))
  return data || {}
}
function short(value:any, fallback="—",maxChars=180):string {
  if (value===null || value===undefined || value==="") return fallback
  return String(value).slice(0,Math.max(1,Math.min(1200,maxChars)))
}
function clock(seconds:any):string {
  const n=Number(seconds)
  if (!Number.isFinite(n) || n<0) return "--:--"
  return String(Math.floor(n/60)).padStart(2,"0")+":"+String(Math.floor(n%60)).padStart(2,"0")
}
function mb(bytes:any):string {
  const size=Number(bytes)
  return Number.isFinite(size) && size>=0 ? (size/1048576).toFixed(1)+" MiB" : "—"
}

// One visual language for singing, video speech, stem separation and GPU setup.
// Indeterminate downloads never invent a speed, percentage or ETA.
function transferState(task:any):string {
  const state=String(task?.state||"")
  const phase=String(task?.download_phase||task?.phase||task?.stage||"").toLowerCase()
  if (state==="ready" || state==="verified") return "已完成校验"
  if (state==="failed" || state==="interrupted") return state==="failed"?"下载失败":"下载中断"
  if (phase.includes("verif")||phase.includes("sha")||phase.includes("hash")) return "校验文件中"
  if (phase.includes("extract")||phase.includes("install")||phase.includes("dependencies")||phase.includes("venv")) return "准备独立环境"
  if (phase.includes("download") || state==="downloading") return "正在下载"
  if (phase.includes("connect")||phase.includes("resolv")) return "连接下载源"
  return "准备中"
}
function transferCompletedRecently(task:any):boolean {
  const completed=String(task?.state||"")
  if (!(["ready","verified"].includes(completed))) return false
  const when=Number(task?.finished_at||task?.updated_at||0)
  return Number.isFinite(when)&&when>0&&Date.now()/1000-when<=7200
}
function transferStageLabel(info:any):string {
  const stage=String(info?.download_phase||info?.phase||info?.stage||"").toLowerCase()
  if(stage.includes("sha256")||stage.includes("verif")||stage.includes("hash"))return "SHA-256 完整性校验"
  if(stage.includes("cuda")&&stage.includes("download"))return "CUDA 运行库下载"
  if(stage.includes("dependency")||stage.includes("dependenc")||stage.includes("venv")||
    stage.includes("environment")||stage.includes("torch")||stage.includes("installing"))
    return "正在准备独立运行环境"
  if(stage.includes("extract"))return "正在解压校验过的文件"
  if(stage.includes("connecting")||stage.includes("resolv")||stage.includes("upstream"))return "正在连接并核对下载源"
  if(stage.includes("download"))return "正在传输模型文件"
  if(stage.includes("trying_next")||stage.includes("retry"))return "正在切换备用下载源"
  if(stage.includes("ready")||stage.includes("verified"))return "已完成下载及校验"
  return transferState(info)
}
function transferActive(task:any):boolean {
  return ["running","queued","downloading","verifying","checking"].includes(String(task?.state||""))
}
function displayScore(value:any):string {
  // 2026-11-13：评分为空时显示「—」，不要再冒充 0.00
  if (value === null || value === undefined || value === "") return "—"
  const n = Number(value)
  if (!Number.isFinite(n)) return "—"
  return n.toFixed(2)
}
function displayRate(value:any):string {
  const n=Number(value)
  if (!Number.isFinite(n)||n<=0) return "测量中"
  return n>=1048576?(n/1048576).toFixed(2)+" MiB/s":(n/1024).toFixed(1)+" KiB/s"
}
function transferEta(value:any):string {
  const seconds=Number(value)
  if (!Number.isFinite(seconds)||seconds<0) return "估算中"
  if(seconds>=3600)return Math.floor(seconds/3600)+"时"+Math.ceil(seconds%3600/60)+"分"
  if(seconds>=60)return Math.floor(seconds/60)+"分"+Math.ceil(seconds%60)+"秒"
  return Math.ceil(seconds)+"秒"
}
function DownloadProgress(props:{title:string,task:any,size?:number,accent?:string,onRetry?:()=>void}) {
  const info:any=props.task||{}
  const phase=transferState(info)
  const verifying=phase==="校验文件中"
  const finished=phase==="已完成校验"
  const active=transferActive(info)
  const error=["failed","interrupted"].includes(String(info.state||""))
  const bytes=Math.max(0,Number(info.bytes_received ?? info.received_bytes ?? 0)||0)
  const total=Number(info.total_bytes ?? info.expected_bytes_estimate ?? props.size ?? 0)
  const knownTotal=Number.isFinite(total)&&total>0
  const rawPct=verifying ? Number(info.verification_pct) : Number(info.transfer_pct ?? info.progress_pct ?? info.pct)
  const pct=finished?100:Number.isFinite(rawPct)&&rawPct>=0?Math.min(99,rawPct):
    (knownTotal&&bytes>0?Math.min(99,bytes/total*100):null)
  const src=String(info.source_name||info.source||"")
  const last=Number(info.last_progress_at||info.updated_at||0)
  const possiblyStuck=active&&(phase==="正在下载"||phase==="连接下载源")&&
    (info.stalled===true||(last>0&&Date.now()/1000-last>35))
  const elapsedFresh=last>0 && Date.now()/1000-last<=35
  const speed=elapsedFresh&&!possiblyStuck ? displayRate(info.speed_bytes_s):"等待数据"
  const eta=elapsedFresh&&!possiblyStuck ? transferEta(info.eta_seconds):"计算中"
  const accent=props.accent||"#60a5fa"
  return <div style={{padding:"15px",border:"1px solid #719bc644",borderRadius:"17px",
    background:"linear-gradient(132deg,rgba(63,104,163,.12),rgba(70,158,205,.045))",
    display:"grid",gap:"12px",minWidth:0}}>
    <div style={{display:"flex",gap:"12px",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap"}}>
      <div style={{display:"flex",gap:"10px",alignItems:"center"}}>
        <div style={{width:"36px",height:"36px",borderRadius:"11px",background:accent+"23",color:accent,
          display:"flex",justifyContent:"center",alignItems:"center",fontWeight:800}}>↓</div>
        <div><div style={{fontWeight:750,fontSize:"14px"}}>{props.title}</div>
        <div style={{fontSize:"12px",opacity:.72}}>{possiblyStuck?"超过 35 秒无数据进展 · 可能卡住":transferStageLabel(info)}</div></div>
      </div>
      <div style={{fontSize:"23px",fontWeight:850,fontVariantNumeric:"tabular-nums",color:error?"#ee8f89":accent}}>
        {pct!==null?(info.progress_estimated===true?"≈":"")+pct.toFixed(1)+"%":"计算中"}
      </div>
    </div>
    <div role="progressbar" aria-label={props.title+"下载进度"} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={pct===null?undefined:Number(pct.toFixed(1))}
      style={{height:"10px",borderRadius:"999px",background:"rgba(119,154,200,.17)",overflow:"hidden"}}>
      <div style={{height:"100%",width:(pct===null?"22%":Math.max(0,pct)+"%"),
        borderRadius:"999px",background:error?"#ed8585":
         `linear-gradient(90deg,${accent},#5eead4)`,opacity:pct===null?.55:1,
        transition:"width 650ms ease"}}/>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(145px,1fr))",gap:"8px"}}>
      {[{label:finished?"模型文件":"已下载 / 总量",value:finished&&bytes===0?"已通过本地校验":
          mb(bytes)+" / "+(knownTotal?mb(total):"未知")},
        {label:"实时速度",value:finished?"下载已结束":verifying?"正在校验":speed},
        {label:"预计剩余",value:finished?"已完成":verifying?"校验中":eta}].map((x:any,i:number)=><div key={i}
        style={{padding:"9px",borderRadius:"10px",background:"rgba(96,139,186,.075)",minWidth:0}}>
          <div style={{fontSize:"10px",opacity:.72}}>{x.label}</div>
          <div style={{fontSize:"12px",fontWeight:650,overflowWrap:"anywhere",marginTop:"5px"}}>{x.value}</div>
        </div>)}
    </div>
    {Number.isFinite(Number(info.transfer_pct)) && Number(info.transfer_pct)>=0 && !finished?
       <div style={{display:"flex",alignItems:"center",gap:"9px",fontSize:"11px",opacity:.8}}>
         <span>当前文件</span>
         <div style={{flex:1,height:"5px",borderRadius:"5px",background:"rgba(119,154,200,.17)",overflow:"hidden"}}>
           <div style={{width:Math.max(0,Math.min(99,Number(info.transfer_pct)))+"%",height:"100%",
             background:accent,borderRadius:"5px"}}/>
         </div>
         <span>{Math.min(99,Math.max(0,Number(info.transfer_pct))).toFixed(1)}%</span>
       </div>:null}
    {src?<div style={{fontSize:"11px",opacity:.70}}>来源：{short(src,"",85)}</div>:null}
    {possiblyStuck?<div style={{fontSize:"12px",color:"#d88b3c"}}>
      请先检查网络；若任务失败，可直接点该模型的「继续下载」，保留有效断点。
    </div>:null}
    {error?<div style={{display:"flex",gap:"9px",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap"}}>
      <div style={{fontSize:"12px",color:"#d88b3c",flex:1}}>
        {short(info.message||info.error||"任务中断，已保存现有模型文件", "任务失败",150)}</div>
      {props.onRetry?<Button onClick={props.onRetry}>确认后重试 →</Button>:null}
    </div>:null}
    {info.progress_estimated===true&&!finished?<div style={{fontSize:"10px",opacity:.6}}>
      百分比为本地字节估算；校验通过后才算安装完成。
    </div>:null}
  </div>
}
function ModelSpotlight(props:{kind:string,icon:string,title:string,subtitle:string,model:string,
  selected:string,device:string,accent:string}) {
  return <div style={{borderRadius:"18px",padding:"18px",
      background:`linear-gradient(125deg,${props.accent}1e,rgba(77,108,152,.05) 75%)`,
      border:`1px solid ${props.accent}42`,display:"grid",gap:"10px"}}>
    <div style={{display:"flex",alignItems:"center",gap:"12px"}}>
      <div style={{width:"44px",height:"44px",borderRadius:"14px",display:"grid",placeItems:"center",
        background:props.accent+"25",color:props.accent,fontSize:"23px"}}>{props.icon}</div>
      <div style={{minWidth:0,flex:1}}>
        <div style={{fontSize:"10px",letterSpacing:"1.3px",color:props.accent,fontWeight:800}}>{props.kind}</div>
        <div style={{fontSize:"18px",fontWeight:800}}>{props.title}</div>
        <div style={{fontSize:"12px",opacity:.7,marginTop:"3px"}}>{props.subtitle}</div>
      </div>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"repeat(3,minmax(0,1fr))",gap:"8px"}}>
      {[{label:"模型",value:props.model},{label:"首选档位",value:props.selected},
        {label:"运行设备",value:props.device}].map((x:any,i:number)=><div key={i}
        style={{padding:"9px",borderRadius:"10px",background:"rgba(121,147,183,.095)",minWidth:0}}>
          <div style={{fontSize:"11px",opacity:.7}}>{x.label}</div>
          <div style={{fontSize:"12px",fontWeight:700,overflowWrap:"anywhere",marginTop:"4px"}}>{x.value}</div>
        </div>)}
    </div>
  </div>
}
function ModelTiles(props:{tiers:any[],states:any,chosen:string,accent:string,
  bytesField?:string,gpuField?:string}) {
  return <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(158px,1fr))",gap:"10px"}}>
    {props.tiers.map((tier:any)=>{
      const item:any=props.states[tier.value]||{}
      const installed=item.state==="verified"||item.verified===true||item.present===true
      const downloading=transferActive(item)||transferActive(item.download)
      const active=tier.value===props.chosen
      const size=item.download_bytes_estimate||item.expected_bytes_estimate||item.plan?.download_bytes_estimate
      const gpu=item.suggested_free_vram_mib||item.vram_estimate_mib||item.plan?.suggested_free_vram_mib
      return <div key={tier.value} style={{padding:"12px",minWidth:0,borderRadius:"14px",
          background:active?props.accent+"12":"rgba(115,153,187,.05)",
          border:active?`1px solid ${props.accent}77`:"1px solid #8198b32b"}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"4px"}}>
          <div style={{fontWeight:850,fontSize:"15px"}}>{tier.label}</div>
          <span style={{fontSize:"10px",padding:"3px 6px",borderRadius:"20px",
            background:installed?"#22a88925":downloading?"#4c93da25":"#80808017",
            color:installed?"#2bb898":props.accent}}>
            {installed?"已安装":downloading?"下载中":"未安装"}</span>
        </div>
        <div style={{fontSize:"12px",fontWeight:700,marginTop:"10px",minHeight:"32px"}}>{tier.name}</div>
        <div style={{fontSize:"11px",opacity:.72,marginTop:"6px"}}>{mb(size)} · 显存建议 {gpu||"—"} MiB</div>
        {downloading?<div style={{display:"grid",gap:"6px",marginTop:"10px"}}>
          <div style={{height:"5px",background:"#799ab62a",borderRadius:"20px",overflow:"hidden"}}>
            <div style={{width:Math.max(1,Math.min(99,Number(item.download?.progress_pct??item.progress_pct??0)))+"%",
              height:"100%",background:props.accent,borderRadius:"20px"}}/>
          </div>
          <div style={{fontSize:"11px",color:props.accent,fontWeight:750}}>{
            Number.isFinite(Number(item.download?.progress_pct??item.progress_pct))
              ? Math.min(99,Math.max(0,Number(item.download?.progress_pct??item.progress_pct))).toFixed(1)+"%"
              : "准备模型中"}</div>
        </div>:null}
      </div>
    })}
  </div>
}

export default function WeiTaSuoTing(props:PluginSurfaceProps) {
  const state:any = props.state || {}
  const status:any = state.status || {}
  const lyrics:any = state.lyrics || {}
  const singerRuntime:any = lyrics.singing_pipeline || status.singing_pipeline || {}
  const catalog:any = state.catalog || {}
  const audio:any = state.audio || {}
  const singing:any = state.singing || {}
  const singingEnvironment:any = singing.environment || {}
  const environmentRemoval:any = singing.environment_removal || {}
  const singingInstalling = singing.preparation?.state === "running"
  const singingRemoving = environmentRemoval.state === "running"
  const resources:any = status.resource_management || {}
  const gpu:any = resources.gpu || {}
  const resourceModels:any = resources.models || {}
  const cpuPermissions:any = resources.cpu_consent || {}
  const groove:any = state.live2d || {}
  const grooveLive:any = groove.live || {}
  const grooveBridge:any = groove.bridge || {}
  const grooveClient:any = grooveBridge.client || {}
  const groovePct=(value:any)=>Math.round(Math.max(0,Math.min(1,Number(value||0)))*100)
  const reserveMiB:number = Number(resources.reserve_mib || 2048)
  const [tab,setTab] = useState<string>("listen")
  const [quality,setQuality] = useState<string>("medium")
  const [device,setDevice] = useState<string>("cpu")
  const [language,setLanguage] = useState<string>("auto")
  const [source,setSource] = useState<string>("mirror")
  const [hasPicked,setHasPicked] = useState<boolean>(false)
  // Historical lyrics are an explicit view choice, never an automatic fallback
  // while a new song is waiting for its first ASR result.
  const [historySongId,setHistorySongId] = useState<string>("")
  const [busy,setBusy] = useState<string>("")
  const [notice,setNotice] = useState<string>("")
  const [error,setError] = useState<string>("")
  const [diagnostics,setDiagnostics] = useState<any>({})
  const [showLog,setShowLog] = useState<boolean>(false)
  // Only an explicit button in the local surface can authorize model downloads
  // or removal. Closing this dialog has zero backend side effects.
  const [dialog,setDialog] = useState<any>(null)
  const [appList,setAppList] = useState<any[]>([])
  const [appFilter,setAppFilter] = useState<string>("")
  const [appsError,setAppsError] = useState<string>("")
  const [audioOutputs,setAudioOutputs] = useState<any[]>([])
  const [outputsError,setOutputsError] = useState<string>("")
  const [advancedOutputs,setAdvancedOutputs] = useState<boolean>(false)
  const [stemQuality,setStemQuality] = useState<string>("ultra")
  const [singingQuality,setSingingQuality] = useState<string>("medium")
  const [singingDevice,setSingingDevice] = useState<string>("cpu")
  const [preview,setPreview] = useState<any>(null)
  const [previewBusy,setPreviewBusy] = useState<string>("")
  const [healthDialogSeen,setHealthDialogSeen] = useState<string>("")

  const backendBuild=String(status.ui_protocol_build || catalog.ui_protocol_build || "")
  const mismatchedBuild=backendBuild !== UI_PROTOCOL_BUILD
  const activeQuality = String(catalog.active_quality || status.lyrics_quality || "")
  useEffect(()=>{
    if (!hasPicked && qualities.some(q=>q.value===activeQuality)) setQuality(activeQuality)
    if (!hasPicked && ["cpu","cuda"].includes(String(catalog.active_device))) setDevice(catalog.active_device)
  },[activeQuality,catalog.active_device,hasPicked])
  useEffect(()=>{
    if (languages.some(lang=>lang.value===status.lyrics_language)) setLanguage(status.lyrics_language)
  },[status.lyrics_language])
  // A new track always returns to the current view. History is only displayed
  // by an explicit click and never carries over into another song session.
  useEffect(()=>{setHistorySongId("")},[lyrics.current_song_id])
  const anyDownloadActive=Boolean(
    (singing.preparation?.state==="running")||
    (singing.environment_removal?.state==="running")||
    Object.values(catalog.models||{}).some((m:any)=>transferActive(m))||
    (status.stem_separation?.download?.state==="running")||
    (status.stem_separation?.gpu_runtime?.install?.state==="running")||
    (status.environment?.model_download?.state==="running"))
  useEffect(()=>{
    // Context contains read-only local status, model progress, and two-song lyrics.
    // No model inference, downloads, or permissions are triggered by refresh.
    let cancelled=false
    let refreshing=false
    const timer=setInterval(()=>{
      if (cancelled || refreshing) return
      refreshing=true
      void props.api.refresh().catch(()=>{}).finally(()=>{refreshing=false})
    },anyDownloadActive?1800:5000)
    return ()=>{cancelled=true;clearInterval(timer)}
  },[props.api,anyDownloadActive])

  const perform=(id:string,args:any={},fallback="已提交")=>{
    if (busy) return
    setBusy(id);setError("");setNotice("")
    // The call begins in a genuine click/change handler. The host is in charge
    // of permission and action dispatch; only @ui.action entries are invoked.
    void props.api.call(id,args,{userInitiated:true})
      .then(unpack)
      .then((out:any)=>{
        setNotice(short(out.note || out.message || fallback))
        return props.api.refresh()
      })
      .catch((e:any)=>setError(short(e?.message || e,"操作失败")))
      .finally(()=>setBusy(""))
  }
  const paused=Boolean(status.plugin_paused)
  const liveState=String((status.live||{}).status || status.capture_recovery?.state || 'not_running')
  const listening=Boolean(status.monitor_enabled && !paused && liveState==='listening')
  const starting=Boolean(status.monitor_enabled && !paused && ['starting','recovering'].includes(liveState))
  const waitingSavedApp=liveState==='waiting_for_application'
  const autoReconnecting=['rebound_waiting_to_start','recovering'].includes(liveState)
  const model=qualities.find(q=>q.value===quality) || qualities[1]
  const item=(catalog.models || {})[model.model] || {}
  const verified=item.state==="verified" && item.sha256_verified === true
  const loading=item.state==="downloading" || item.state==="verifying"
  const failed=item.state==="failed" || item.state==="interrupted"
  // An explicit zero from a fresh restart MUST beat an old ~99% estimate.
  const rawPct=(item.progress_pct!==undefined && item.progress_pct!==null)
    ? Number(item.progress_pct)
    : Math.floor(100*Number(item.received_bytes||0)/Math.max(1,Number(item.expected_bytes_estimate||1)))
  const percent=Math.max(0,Math.min(99,Number.isFinite(rawPct)?rawPct:0))
  const switchState=catalog.switch || status.lyrics_switch || {}
  const switchBusy=switchState.status==="preparing"
  const songs:any[]=Array.isArray(lyrics.songs)?lyrics.songs:[]
  const activeSongId=String(lyrics.current_song_id || "")
  const lookingAtHistory=Boolean(historySongId && historySongId!==activeSongId &&
    songs.some((s:any)=>String(s.id)===historySongId))
  const current=lookingAtHistory
    ? songs.find((s:any)=>String(s.id)===historySongId)
    : activeSongId ? songs.find((s:any)=>String(s.id)===activeSongId) : null
  // Do not synthesize a current lyric line from a stale, previous-song row.
  const cues:any[]=Array.isArray(current?.cues)?current.cues.filter((c:any)=>c?.text_preview):[]
  const timeline=current?.recognition_progress || {}
  const env=status.environment || {}
  const live=status.live || {}
  const appCapture:any = status.app_capture || {}
  const commentary:any = status.commentary || {}
  // Server meter monitor is fresher than the last manual scan. Never surface
  // process inventory as sound activity; only these peaks count as audible.
  const liveApps:any[] = Array.isArray(appCapture.audible_sources) &&
    Number(appCapture.last_scanned_at||0)>0
    ? appCapture.audible_sources : appList
  const selectedApp:any = appCapture.selected || {}
  const appCaptureInstaller:any = appCapture.installer || {}
  const appModeNames:Record<string,string> = {monitor:"仅监测",music:"音乐识别",video:"视频语音",auto:"自动判断音乐",ai:"AI语音不转写"}
  const scene:any=status.scene||{}
  const mediaSources:any[]=Array.isArray(status.media_sources)?status.media_sources:[]
  const sceneNames:Record<string,string>={music:"音乐",video:"视频声音",mixed:"多音源混合 · 暂停识别",
    ai_output:"AI 自己的声音 · 暂停识别",unknown:"音源待确认"}
  const sceneName=sceneNames[scene.kind]||"等待声音"
  // The card below is a status panel, not a fixed reminder: show why the
  // current class was chosen instead of a permanent "awaiting review" label.
  const sceneReasonNames:Record<string,string>={
    not_started:"尚未开始判定（先让所选应用出声）",
    app_not_selected:"还没有选择要单独采集的应用",
    awaiting_ai_or_window_label_review:"标签仍是候选，等待窗口/AI 复核",
    label_and_audio_scene_conflict:"复核标签与本地声学证据不一致，已暂停识别",
    reviewed_video_but_channel_silent:"视频已复核，但此刻声道是安静的",
    reviewed_music_but_channel_silent:"音乐已复核，但此刻声道是安静的",
    selected_app_channel_silent:"声道安静，没有可判定的声音",
    selected_video_no_speech_evidence:"视频声道里没有检测到人声",
    video_audio_music_only_no_speech:"只有音乐/歌声，暂无人声（模型不启动）",
    music_video_music_audio_channel:"音乐视频：画面是视频、声音是音乐，已分道识别音乐",
    reviewed_video_speech_recent_evidence:"刚听到人声，正在等下一段对话（12 秒内保持可用）",
    reviewed_video_speech_with_background_music:"视频人声（可含背景音乐）",
    user_selected_video_source_with_speech_evidence:"视频人声已确认",
    selected_app_not_music:"按音乐模式判定，但这窗不像音乐",
    selected_app_music_confirmed:"音乐已确认",
    system_capture_selected:"当前是系统混音模式",
  }
  const videoCues:any[] = Array.isArray(audio.video_speech_candidates)?audio.video_speech_candidates:[]
  const sceneVerified=scene.verified===true
  const stem:any=status.stem_separation || {}
  const stemDownload:any=stem.download || {}
  const stemGpu:any=stem.gpu_runtime || {}
  const gpuInstall:any=stemGpu.install || {}
  const gpuProbe:any=stemGpu.probe || {}
  const stemProgressActive=stemDownload.running===true && stemDownload.state==="running"
  const stemVerified=stemDownload.sha256_verified===true &&
    stemDownload.model===stem.quality && stem.model_present===true
  const stemModels:any=stem.models || {}
  const health:any=status.automatic_health || {}
  const stemPlan:any=catalog.stem_plan || {}
  const plans:any=(catalog.models||{})[(qualities.find(q=>q.value===String(dialog?.quality))||qualities[1]).model]?.plan || {}
  const dialogIsLyrics=dialog?.kind==="lyrics-download" || dialog?.kind==="lyrics-uninstall" || dialog?.kind==="lyrics-activate"
  const dialogIsStem=String(dialog?.kind||"").startsWith("stem-") && dialog?.kind!=="stem-gpu-install"
  const dialogStem:any=stemModels[dialog?.quality||stemQuality] || {}
  const dialogPlan:any=dialogIsLyrics?plans:dialogIsStem?{
    download_bytes_estimate:dialogStem.expected_bytes_estimate||stemPlan.download_bytes_estimate,
    disk_required_bytes_estimate:(dialogStem.expected_bytes_estimate||stemPlan.download_bytes_estimate||0)*2+32000000,
    suggested_free_vram_mib:dialogStem.vram_estimate_mib||stemPlan.suggested_free_vram_mib,
    estimated_gpu_peak:"取决于显卡运行库及具体声音，请以实机诊断和试听为准",
  }:{}
  const dialogIsSinging=String(dialog?.kind||"").startsWith("singing-") &&
     !["singing-environment-uninstall","singing-install-cancel"].includes(String(dialog?.kind||""))
  const singingPlans:any=(singing.models||{})[dialog?.quality||singingQuality]||{}
  const singingName=singingOptions.find(q=>q.value===(dialog?.quality||singingQuality))?.name||"专业歌词"
  const modelDisplay=(qualities.find(q=>q.value===String(dialog?.quality))||qualities[1])
  const gpuEstimate=(domain:string):number=>{
    if (domain==="singing") return Number((singing.models||{})[singingQuality]?.suggested_free_vram_mib || 0)
    if (domain==="video") return Number(item.plan?.suggested_free_vram_mib || 0)
    return Number((stemModels||{})[stemQuality]?.vram_estimate_mib || 0)
  }
  const applyDevice=(domain:string,value:string)=>{
    if (domain==="singing") {setSingingDevice(value);setNotice("已选择歌声运行设备；正式启用需试推理")}
    if (domain==="video") {setDevice(value);setHasPicked(true);setNotice("已选择视频运行设备；正式启用需试推理")}
    if (domain==="stem") perform("set_stem_device",{device:value,confirmed:true},"分离计算设备已保存")
  }
  const chooseDevice=(domain:string,value:string)=>{
    if (value==="cuda" && gpu.available===true && Number.isFinite(gpu.free_mib) &&
        Number(gpu.free_mib)<gpuEstimate(domain)+reserveMiB) {
      setDialog({kind:"gpu-shortage",domain,free_mib:Number(gpu.free_mib),
        estimated_mib:gpuEstimate(domain)})
      return
    }
    applyDevice(domain,value)
  }
  const requestResourcePolicy=(domain:string,kind:string,next:any)=>
    setDialog({kind:"resource-policy",domain,change:kind,...next})
  const onConfirm=()=>{
    if (!dialog || busy) return
    const act=dialog
    setDialog(null)
    if (act.kind==="singing-environment-uninstall") perform("uninstall_singing_environment",
        {confirmed:true},"已开始安全卸载歌声独立 Python 环境；模型文件和其他环境保留")
    else if (act.kind==="singing-install-cancel") perform("cancel_singing_installation",
        {confirmed:true},"正在停止歌声模型下载和依赖安装，请稍候")
    else if (act.kind==="singing-device") chooseDevice("singing",act.value)
    else if (act.kind==="singing-download") perform("install_singing_model",
        {quality:act.quality,source,device:singingDevice,confirmed:true},"正在后台准备独立环境并下载校验歌声模型")
    else if (act.kind==="singing-import") perform("verify_singing_import",
        {quality:act.quality,confirmed:true},"正在后台离线校验手动导入的模型")
    else if (act.kind==="singing-activate") perform("activate_singing_model",
        {quality:act.quality,device:singingDevice},"正在实际加载并预热歌声模型")
    else if (act.kind==="singing-uninstall") perform("uninstall_singing_model",
        {quality:act.quality,confirmed:true},"将安全卸载此专业歌词模型")
    else if (act.kind==="lyrics-download") perform("stage_lyrics_model",
        {quality:act.quality,download_source:source,confirmed:true},"已提交后台下载；验证通过前不会启用")
    else if (act.kind==="lyrics-uninstall") perform("uninstall_lyrics_model",
        {quality:act.quality,confirmed:true,auto_pause:true},"正安全释放并卸载所选档位")
    else if (act.kind==="lyrics-activate") perform("activate_verified_lyrics_model",
        {quality:act.quality,device},"模型试推理中，成功后才会切换")
    else if (act.kind==="stem-download") perform("start_stem_download",
        {confirmed:true,quality:act.quality||stemQuality,download_source:source},"分离模型后台下载已开始")
    else if (act.kind==="stem-uninstall") perform("uninstall_stem_model",
        {confirmed:true,quality:act.quality||stemQuality,auto_pause:true},"已请求安全卸载此档位")
    else if (act.kind==="stem-select") perform("select_stem_quality",
        {confirmed:true,quality:act.quality||stemQuality},"分离模型已切换，旧模型文件保留")
    else if (act.kind==="stem-enable") perform("set_stem_separation",
        {enabled:true},"已请求启用双轨分离")
    else if (act.kind==="device") chooseDevice("video",act.value)
    else if (act.kind==="source") {setSource(act.value);perform("set_download_source",
        {source:act.value,confirmed:true},"下载来源偏好已保存")}
    else if (act.kind==="stem-device") chooseDevice("stem",act.value)
    else if (act.kind==="stem-gpu-install") perform("install_stem_gpu",
        {confirmed:true},"已请求后台安装 NVIDIA 分离加速包")
    else if (act.kind==="resource-policy") perform("set_adaptive_resources",{
        domain:act.domain,enabled:act.enabled, reserve_mib:act.reserve_mib,
        cpu_fallback:act.cpu_fallback, gpu_ram_offload:act.gpu_ram_offload,
        confirmed:true},"显存自适应与内存授权已更新")
    else if (act.kind==="gpu-shortage") applyDevice(act.domain,"cpu")
    else if (act.kind==="health-error") setDialog(null)
    else if (act.kind==="media-role") perform("set_media_source_role",
        {app:act.app,role:act.role},"音源身份已更新")
    else if (act.kind==="app-install") perform("install_app_capture",
        {confirmed:true},"已开始安装逐应用采集组件")
    else if (act.kind==="app-select") {
      const args={pid:act.pid,created:act.created,mode:act.mode,confirmed:true}
      if (listening) perform("focus_live_audio_source",args,
          "已安全切换到指定应用声道，候选标签待复核")
      else perform("select_app_capture",args,
          "已选定指定应用声道；开启监听后开始采集")
    }
    else if (act.kind==="pin-host") perform("pin_ai_main_process",
        {pid:act.pid,created:act.created,confirmed:true},
        "已固定 AI 主程序，请重新扫描有声应用")
    else if (act.kind==="app-system") perform("select_system_capture",
        {confirmed:true},"已恢复原扬声器混合音频采集")
    else if (act.kind==="channel-share") perform("set_external_channel_share",
        {enabled:act.enabled,confirmed:true},"已更新其他插件的声音摘要读取权限")
    else if (act.kind==="capture-output") perform("set_capture_output",
        {output_id:act.output_id,ai_separate:act.ai_separate===true,confirmed:true},
        "已保存监听设备。恢复插件前请在 Windows 音量合成器确认路由")
  }
  useEffect(()=>{
    if (["mirror","official"].includes(status.download_source_preference))
      setSource(status.download_source_preference)
  },[status.download_source_preference])
  useEffect(()=>{
    if (stemChoices.some(q=>q.value===stem.quality)) setStemQuality(stem.quality)
  },[stem.quality])
  useEffect(()=>{
    const token=String(health.run_id||0)+"|"+String(health.error||"")
    if (health.state==="failed" && token && token!==healthDialogSeen) {
      setHealthDialogSeen(token)
      setDialog({kind:"health-error",error:String(health.error||"")})
    }
  },[health.state,health.error,health.run_id,healthDialogSeen])
  const previewSound=(stage:string)=>{
    if(previewBusy || busy) return
    setPreviewBusy(stage);setError("")
    void props.api.call("get_stem_preview",{stage},{userInitiated:true})
      .then(unpack).then((result:any)=>{
        if (!result.data_uri || !String(result.data_uri).startsWith("data:audio/wav;base64,"))
          throw Error("试听音频不可用")
        setPreview({stage,uri:String(result.data_uri),duration:result.duration_s,
                    model:result.model,strength:result.strength})
      }).catch((e:any)=>setError(short(e?.message||e,"试听失败")))
        .finally(()=>setPreviewBusy(""))
  }

  const transferJobs:any[]=[]
  const showRecent=(s:any)=>transferCompletedRecently(s)
  if (singing.preparation?.state==="running"||singing.preparation?.state==="failed"||singing.preparation?.state==="interrupted"||
      showRecent(singing.preparation)) {
    const tier=String(singing.preparation.quality||singingQuality)
    transferJobs.push({key:"singing-"+tier,title:"专业歌声 · "+(tierLabels[tier]||"模型"),
      info:singing.preparation,size:(singing.models||{})[tier]?.download_bytes_estimate,accent:"#8b80ef",
      onRetry:()=>setDialog({kind:"singing-download",quality:tier})})
  }
  for (const q of qualities) {
    const m:any=(catalog.models||{})[q.model]||{}
    if (["downloading","verifying","checking","interrupted","failed"].includes(String(m.state||"")) &&
         (m.download_phase||m.download_active||m.prepare_failed))
      transferJobs.push({key:"video-"+q.value,title:"视频语音 · "+q.label+" · Whisper "+q.name,
        info:m,size:m.expected_bytes_estimate||m.plan?.download_bytes_estimate,accent:"#50a2ee",
        onRetry:()=>setDialog({kind:"lyrics-download",quality:q.value})})
  }
  const lastVideo=status.lyrics_model_preparation||{}
  if (showRecent(lastVideo)) {
    const chosen=qualities.find(q=>q.model===lastVideo.model)
    if (chosen && !(transferJobs.some(job=>job.key==="video-"+chosen.value)))
      transferJobs.push({key:"video-"+chosen.value,
        title:"视频语音 · "+chosen.label+" · Whisper "+chosen.name,
        info:{...lastVideo,progress_pct:100},
        size:(catalog.models||{})[chosen.model]?.expected_bytes_estimate||
          (catalog.models||{})[chosen.model]?.plan?.download_bytes_estimate,
        accent:"#50a2ee",onRetry:()=>setDialog({kind:"lyrics-download",quality:chosen.value})})
  }
  if (stemDownload.state==="running"||stemDownload.state==="failed"||stemDownload.state==="interrupted"||showRecent(stemDownload))
    transferJobs.push({key:"stem",title:"人声 / 伴奏分离 · "+(tierLabels[stemDownload.model]||"模型"),
      info:stemDownload,size:stemModels[stemDownload.model]?.expected_bytes_estimate,accent:"#33bea9",
      onRetry:()=>setDialog({kind:"stem-download",quality:stemDownload.model||stemQuality})})
  if (gpuInstall.state==="running"||gpuInstall.state==="failed"||gpuInstall.state==="interrupted"||showRecent(gpuInstall))
    transferJobs.push({key:"gpu",title:"私有 CUDA / cuDNN 运行环境",info:gpuInstall,
      size:gpuInstall.total_bytes||stemGpu.download_bytes_estimate,accent:"#f9a36e",
      onRetry:()=>setDialog({kind:"stem-gpu-install"})})
  if (env.model_download?.state==="running"||env.model_download?.state==="failed"||showRecent(env.model_download))
    transferJobs.push({key:"yamnet",title:"YAMNet · 基础音乐分类",info:env.model_download,
      size:env.model_download.total_bytes,accent:"#33bcbd"})

  // Hosted TSX places Page content in an independently scrolling container.
  // Every confirmation MUST be a sibling to Page (not nested inside it),
  // and use the host's supported Modal instead of an ad-hoc fixed overlay.
  // Otherwise scrolling/transformed Page ancestors change the containing block
  // and the popup can slide outside the visible viewport even with huge z-index.
  const dialogTitle = dialog ? (dialog.kind==="singing-download"?"安装专业歌声模型":
            dialog.kind==="singing-activate"?"验证并启用专业歌声模型？":
            dialog.kind==="singing-uninstall"?"卸载选中的专业歌词模型？":
            dialog.kind==="singing-environment-uninstall"?"卸载专业歌声独立环境？":
            dialog.kind==="singing-install-cancel"?"停止当前模型安装？":
            dialog.kind==="singing-import"?"核验手动下载的专业歌声模型？":
            dialog.kind==="singing-device"?"选择专业歌声计算设备？":
            dialog.kind==="resource-policy"?"确认显存自适应策略？":
            dialog.kind==="gpu-shortage"?"显存不足：仍要尝试 GPU 吗？":
            dialog.kind==="lyrics-download"?"下载视频语音模型":
            dialog.kind==="lyrics-uninstall"?"确定卸载此档位？":
            dialog.kind==="lyrics-activate"?"验证并启用模型？":
            dialog.kind==="stem-download"?"下载安装人声分离模型？":
            dialog.kind==="stem-uninstall"?"安全卸载分离模型":
            dialog.kind==="stem-select"?"切换分离模型":
            dialog.kind==="health-error"?"自动检测发现问题":
            dialog.kind==="stem-enable"?"开启双轨人声分离？":
            dialog.kind==="stem-gpu-install"?"安装 NVIDIA 人声分离加速包？":
            dialog.kind==="stem-device"||dialog.kind==="device"?"选择计算设备？":
            dialog.kind==="media-role"?"确认音源身份？":
            dialog.kind==="app-install"?"安装应用声道采集组件？":
            dialog.kind==="app-select"?"只监听指定应用？":
            dialog.kind==="pin-host"?"固定这个程序为 AI 声音来源？":
            dialog.kind==="app-system"?"恢复混合扬声器采集？":
            dialog.kind==="channel-share"?"授权其他插件读取选中声道？":
            dialog.kind==="capture-output"?"切换监听扬声器":"切换下载来源") : "确认操作"
  const ob = (state && state.onboarding) || {}
  const [showGuide,setShowGuide] = useState<boolean>(false)
  const [guideStep,setGuideStep] = useState<number>(1)
  const needGuide = Boolean(ob.first_run && !ob.completed) || showGuide
  if (needGuide) {
    return <Page title="为她所听" subtitle="先带她认识你的耳朵">
      <Onboarding ob={ob} api={props.api} startStep={guideStep} onClose={()=>setShowGuide(false)}/>
    </Page>
  }
  const missingCount = (ob.scan && Array.isArray(ob.scan.items))
    ? ob.scan.items.filter((it:any)=>it && it.required && it.state!=="ok").length : 0
  return <>
  <Page title="为她所听" subtitle="全应用音频感知 · 资源自动调度">
    <div style={{borderRadius:"22px",padding:"clamp(16px,2.8vw,26px)",marginBottom:"14px",
      color:"#f1f7ff",border:"1px solid #5689c547",
      background:"linear-gradient(126deg,#112c4b 0%,#153a56 53%,#10525b 100%)",
      boxShadow:"0 12px 28px rgba(22,58,91,.12)"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"14px",flexWrap:"wrap"}}>
        <div style={{display:"flex",alignItems:"center",gap:"12px"}}>
          <div style={{width:"48px",height:"48px",borderRadius:"15px",display:"grid",placeItems:"center",
            fontSize:"25px",color:"#8ef6f0",background:"#4b9eaf2d"}}>♫</div>
          <div><div style={{fontSize:"11px",letterSpacing:"2.1px",color:"#8df0ea",fontWeight:800}}>
            AUDIO INTELLIGENCE</div>
            <div style={{fontSize:"21px",fontWeight:850,marginTop:"2px"}}>为她所听 · 音频中控</div></div>
        </div>
        <span style={{fontSize:"11px",padding:"7px 11px",borderRadius:"99px",background:"#ffffff17",
             border:"1px solid #ffffff32",color:"#cde6f8"}}>{paused?"已暂停":listening?"● 正在监听":"● 待机"}</span>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,minmax(0,1fr))",gap:"9px",marginTop:"18px"}}>
        {[{label:"当前音源",value:short(sceneName,"待确认",28)},
          {label:"内容类型",value:scene.kind==="video"?"视频语音":scene.kind==="music"?"音乐":"待确认 / 仅监测"},
          {label:"运行档位",value:scene.kind==="video"?(tierLabels[activeQuality]||"未启用"):
            (tierLabels[singing.active_quality]||"未启用")}].map((metric:any,i:number)=><div key={i}
            style={{borderRadius:"11px",background:"#ffffff0b",border:"1px solid #ffffff17",padding:"10px",minWidth:0}}>
              <div style={{fontSize:"11px",color:"#adcee1"}}>{metric.label}</div>
              <div style={{fontSize:"13px",marginTop:"6px",fontWeight:750,overflowWrap:"anywhere"}}>{metric.value}</div>
            </div>)}
      </div>
    </div>
    {missingCount>0?<Alert tone="warning">
      有 {missingCount} 项还没装好
      <Button onClick={()=>{setGuideStep(2);setShowGuide(true)}}>去补齐</Button>
    </Alert>:null}
    <div style={{padding:"8px 0"}}><SegmentedControl value={tab} options={tabs} onChange={setTab}/></div>
    {mismatchedBuild?<Alert tone="warning">版本未同步：界面 {UI_PROTOCOL_BUILD}，后台 {backendBuild || "旧版或未响应"}。先点本页的重新加载按钮；仍不一致就完全退出并重启 N.E.K.O。不要通过插件中心重新安装，那会覆盖本地补丁。</Alert>:null}
    {error?<Alert tone="warning">{error}</Alert>:null}
    {notice?<Alert tone="info">{notice}</Alert>:null}
    {tab!=="model" && transferJobs.some(job=>transferActive(job.info))?<div
      style={{display:"flex",gap:"12px",alignItems:"center",justifyContent:"space-between",
        flexWrap:"wrap",borderRadius:"15px",padding:"12px 16px",marginBottom:"13px",
        border:"1px solid #5496bc59",background:"rgba(66,133,179,.095)"}}>
      <div style={{display:"grid",gap:"4px"}}>
        <div style={{fontWeight:750,fontSize:"13px"}}>↓ 模型下载任务正在运行</div>
        <div style={{opacity:.76,fontSize:"11px"}}>正在记录速度、百分比与卡住状态，不需要关闭其他界面。</div>
      </div>
      <Button onClick={()=>setTab("model")}>查看实时进度 →</Button>
    </div>:null}

    {tab==="listen"? <>
      <Card title="电脑音频感知"><Stack>
        <StatusBadge tone={paused?"warning":listening?"success":"neutral"}>
          {paused?"已暂停 · 释放插件显存":listening?"监听中":waitingSavedApp?
           "等待原应用 · 自动守候":autoReconnecting?
           "采集断线 · 自动恢复中":starting?"正在启动":
            status.monitor_enabled?"监听未运行 · 请检查设备":"等待授权"}
        </StatusBadge>
        <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
          {!listening && !paused && !starting && !waitingSavedApp && !autoReconnecting?<Button disabled={Boolean(busy)} onClick={()=>perform("start_auto_listen",{consent:true},"已申请启动声音监听")}>{status.monitor_enabled?"重试监听":"开启监听"}</Button>:null}
          {listening?<Button disabled={Boolean(busy)} onClick={()=>perform("pause_plugin",{},"已暂停插件并释放歌词模型显存")}>暂停 / 释放显存</Button>:null}
          {paused?<Button disabled={Boolean(busy)} onClick={()=>perform("resume_plugin",{},"已恢复插件")}>恢复插件</Button>:null}
          {status.monitor_enabled?<Button disabled={Boolean(busy)} onClick={()=>perform("stop_auto_listen",{},"已停止监听")}>关闭监听</Button>:null}
        </div>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"12px"}}>
          <div><Text>AI 私有感知</Text>
            <Text>静默提交，不直接发聊天消息。</Text></div>
          <Switch checked={Boolean(status.ai_delivery_enabled)} disabled={Boolean(busy)}
            onChange={(on:boolean)=>perform("set_ai_delivery",{enabled:on},on?"AI 私有感知已开启":"AI 私有感知已暂停")}/>
        </div>
        <Text>当前音源：{appCapture.mode==="application"?short(selectedApp.name,"所选应用"):short(live.device_name||live.device,"系统混合扬声器")}</Text>
        <Text>{sceneName} · {sceneVerified && scene.tag_status==="external_review"?
          "AI / 窗口已复核，允许按需识别":"候选音源等待 AI / 窗口复核，暂不加载识别模型"} · 仅供 AI 私有感知</Text>
        <Text>歌声识别：{short(singerRuntime.reason,"等待单一音乐音源",150)} · {singerRuntime.model_loaded_and_tested?
          "模型已加载（"+short(singerRuntime.actual_device,"未验证")+"）":"模型未驻留显存"} · 队列 {Number(singerRuntime.queue_depth||0)}</Text>
        {scene.kind==="mixed"?<Alert tone="warning">多音源混合：歌词、视频语音和 AI 感知暂时停止，声音恢复单一来源后自动继续。</Alert>:null}
        {status.monitor_start_error||live.error?<Alert tone="warning">{short(status.monitor_start_error||live.error)}</Alert>:null}
      </Stack></Card>
      <Card title="当前声音文字"><Stack>
        {scene.kind==="video"?<>
          <Text>视频语音：按原始停顿分段，不按歌曲节拍截断</Text>
          <div style={{maxHeight:"320px",overflowY:"auto",overscrollBehaviorY:"contain"}}>
            {videoCues.length?videoCues.slice(-18).map((cue:any,i:number)=><div key={String(cue.at_s??i)+":"+i}
              style={{display:"flex",gap:"12px",padding:"7px 0",borderBottom:"1px solid #80808028"}}>
              <span style={{opacity:.65,minWidth:"45px"}}>{clock(cue.at_s)}</span>
              <span style={{whiteSpace:"pre-wrap",overflowWrap:"anywhere"}}>
                {cue.silence_boundary?<Text>—— 新语音段落 ——</Text>:null}
                {short(cue.text_candidate,"")}
              </span>
            </div>):<Text>视频声音已确认，等待识别出可靠的语音文字</Text>}
          </div>
        </>:scene.kind==="mixed"||scene.kind==="ai_output"?<Text>来源混合或 AI 自己发声，已暂停本轮识别，不沿用旧文字。</Text>
        :scene.kind!=="music" && !lookingAtHistory?<Text>正在扫描音源，等待 AI / 窗口确认音乐或视频类型；旧歌词不会冒充当前歌曲。</Text>
        :!lookingAtHistory && (!sceneVerified || scene.tag_status!=="external_review")?<Text>需要音源复核后才能识别当前歌曲。AI 可扫描有声应用并复核标签。</Text>:<>
        <Text>{lookingAtHistory?"历史歌词 · 仅供查看":"当前音乐歌词 · 按语义与自然停顿断句"}</Text>
        {songs.length?<div style={{display:"flex",flexWrap:"wrap",gap:"8px"}}>
          <Button onClick={()=>setHistorySongId("")}>当前 · {short(songs.find((s:any)=>String(s.id)===activeSongId)?.title,"等待新歌",18)}</Button>
          {songs.filter((song:any)=>String(song.id)!==activeSongId).slice(-1).map((song:any)=><Button key={String(song.id)}
            onClick={()=>setHistorySongId(String(song.id))}>上一首 · {short(song.title,"未命名",18)}</Button>)}
        </div>:<Text>自动监听已启用，等待经复核的歌曲</Text>}
        {current?<>
          <Text>{short(lyricStates[timeline.state]||timeline.state,"等待识别")} · 已处理 {clock(timeline.completed_until_s||0)}</Text>
          <div style={{maxHeight:"320px",overflowY:"auto"}}>
            {(current?.display_lines||[]).length?(current.display_lines||[]).slice(-18).map((cue:any,i:number)=><div key={String(cue.at_s??i)+":"+i}
              style={{display:"flex",gap:"12px",padding:"7px 0",borderBottom:"1px solid #80808028"}}>
                <span style={{opacity:.65,minWidth:"45px"}}>{clock(cue.player_at_s_estimated??cue.at_s)}</span>
                <span style={{whiteSpace:"pre-wrap",overflowWrap:"anywhere"}}>
                  {short(cue.text_preview,"")}
                  <Text>乐句 {Number(cue.phrase_index||0)+1} · {Number(cue.source_cues||1)} 段语音合并</Text>
                </span>
              </div>):<Text>{short(timeline.last_error||lyrics.recognition_error||"正在等第一句歌词")}</Text>}
          </div>
        </>:null}
        </>}
      </Stack></Card>
      <Card title="吐槽"><Stack>
        <Text>她会像坐在旁边一起听/看的朋友一样随口说一句：歌突然炸开、安静下来、变快变慢、换了一种风格时说说感觉；看视频时挑有意思的地方吐槽。不会报数字。</Text>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"12px"}}>
          <div><Text>音乐吐槽</Text><Text>听到明显变化时说说感受</Text></div>
          <Switch checked={Boolean(commentary.music_on)} disabled={Boolean(busy)}
            onChange={(on:boolean)=>perform("set_commentary",{music:on},on?"音乐吐槽已开启":"音乐吐槽已关闭")}/>
        </div>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"12px"}}>
          <div><Text>视频吐槽</Text><Text>{status.suokong_available?"用「为她所控」看当前视频窗口":
            commentary.screen_share?"用主程序的屏幕分享画面":"需要「为她所控」，或在主程序里打开屏幕分享"}</Text></div>
          <Switch checked={Boolean(commentary.video_on)} disabled={Boolean(busy)}
            onChange={(on:boolean)=>perform("set_commentary",{video:on},on?"视频吐槽已开启":"视频吐槽已关闭")}/>
        </div>
        <SegmentedControl value={String(commentary.frequency||"lively")}
          options={[{value:"lively",label:"活跃 · 约 1.5 分钟"},{value:"normal",label:"适中 · 约 4 分钟"},{value:"rare",label:"很少 · 约 10 分钟"}]}
          onChange={(v:string)=>perform("set_commentary",{frequency:v},"吐槽频率已更新")}/>
        {commentary.took_over_official?<Text>主程序的定时主动搭话已暂停，关掉吐槽后会恢复你原来的设置。</Text>:null}
        {commentary.music_on||commentary.video_on?<Text>{commentary.last&&commentary.last.at?
          "上一次："+(commentary.last.kind==="music"?"听歌 · ":commentary.last.kind==="screen"?"屏幕分享 · ":"看视频 · ")+short(commentary.last.what,"",40)
          :"还没说过"}{commentary.why_not_now==="cooldown"?" · 冷却中":commentary.why_not_now==="user_active"?" · 你刚在聊天，先不打扰":""}</Text>:null}
      </Stack></Card>
      <Text>识别出的文字、节拍和伴奏事件不会直接发送到聊天框；AI 自己决定是否使用与如何表达。开启「吐槽」后，她会在明显变化时主动开口。</Text>
    </>:null}

    {tab==="source"?<>
      <Card title="逐应用声道中控"><Stack>
        <StatusBadge tone={appCapture.mode==="application"?"success":"neutral"}>
          {appCapture.mode==="application"?"单应用独立采集":"系统混音模式"}
        </StatusBadge>
        <Text>{appCapture.mode==="application"?
          (short(selectedApp.name,"未命名应用")+" · PID "+String(selectedApp.pid||0)+" · "+(appModeNames[selectedApp.mode]||"仅监测")):
          "未选择独立应用；原有扬声器混音监听仍可使用"}</Text>
        <Text>只采集你选中的应用及其子进程，不会把整个电脑所有声音同时送去分析。</Text>
        <Text>采集组件：{appCaptureInstaller.ready===true?"已安装":appCaptureInstaller.running?"安装中":"未安装"}</Text>
        {appCaptureInstaller.running?<Text>{short(appCaptureInstaller.message,"正在准备采集组件")}</Text>:null}
        {appCaptureInstaller.state==="failed"?<Alert tone="warning">{short(appCaptureInstaller.message,"安装失败")}</Alert>:null}
        <div style={{display:"flex",gap:"6px",flexWrap:"wrap"}}>
          {!appCaptureInstaller.ready?<Button disabled={Boolean(busy)||appCaptureInstaller.running}
            onClick={()=>setDialog({kind:"app-install"})}>安装独立采集组件</Button>:null}
          <Button disabled={Boolean(busy)}
            onClick={()=>{
              if(busy)return
              setBusy("list-apps");setAppsError("")
              void props.api.call("list_application_sources",{}, {userInitiated:true})
                .then(unpack).then((response:any)=>{
                  if(response.ok===false)throw Error(String(response.error||"扫描失败"))
                  setAppList(Array.isArray(response.apps)?response.apps:[])
                }).catch((e:any)=>setAppsError(short(e?.message||e)))
                .finally(()=>setBusy(""))
            }}>立即扫描有声应用</Button>
          {appCapture.mode==="application"?<Button disabled={Boolean(busy)||listening||starting}
            onClick={()=>setDialog({kind:"app-system"})}>恢复系统混音</Button>:null}
          {!listening&&!starting&&!paused&&appCapture.mode==="application"?<Button disabled={Boolean(busy)}
            onClick={()=>perform("start_auto_listen",{consent:true},"正在监听所选应用")}>开始监听所选应用</Button>:null}
          {paused&&appCapture.mode==="application"?<Button disabled={Boolean(busy)}
            onClick={()=>perform("resume_plugin",{},"已恢复选定应用监听")}>恢复应用监听</Button>:null}
        </div>
        {appsError||appCapture.scan_error?<Alert tone="warning">{short(appsError||appCapture.scan_error)}</Alert>:null}
        {liveApps.length?<>
          <Text>正在发声：{liveApps.length} 个应用 · 音量会话电平检测（不使用模型）</Text>
          <input value={appFilter} placeholder="搜索进程名 / PID" onChange={(e:any)=>setAppFilter(String(e.target.value||""))}
            style={{padding:"8px",borderRadius:"8px",border:"1px solid #80808055",width:"100%"}}/>
          <div style={{maxHeight:"380px",overflowY:"auto",overscrollBehaviorY:"contain",paddingRight:"4px"}}>
          {liveApps.filter((a:any)=>!appFilter||String(a.name||"").toLowerCase().includes(appFilter.toLowerCase())||String(a.pid).includes(appFilter))
            .slice(0,32).map((a:any)=><div key={String(a.pid)+String(a.created)}
            style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"8px",padding:"7px 0",borderBottom:"1px solid #80808022"}}>
            <div style={{display:"flex",flexDirection:"column",gap:"3px",minWidth:0}}>
              <Text>{short(a.name,"未知应用")} · PID {a.pid}</Text>
              <Text>{short(a.label,a.host_ai?"AI 声音（不采集）":"待确认")}
                {a.tag_status==="external_review"?" · 窗口/AI 已复核":a.host_ai?"":" · 待确认"}
                {typeof a.level_dbfs==="number"?" · "+a.level_dbfs+" dBFS":""}</Text>
            </div>
            {a.host_ai?<StatusBadge tone="warning">AI 固定隔离 · 不采集</StatusBadge>:
              <div style={{display:"flex",gap:"4px",flexWrap:"wrap"}}>
                {a.path?<Button disabled={Boolean(busy)||listening||starting}
                  onClick={()=>setDialog({kind:"pin-host",pid:a.pid,created:a.created,name:a.name})}>设为 AI 主程序</Button>:null}
                {[{v:"monitor",t:"监测"},{v:"music",t:"音乐"},{v:"video",t:"视频"}].map((mode:any)=><Button key={mode.v}
                  disabled={Boolean(busy)||!appCaptureInstaller.ready}
                  onClick={()=>setDialog({kind:"app-select",pid:a.pid,created:a.created,name:a.name,mode:mode.v})}>{mode.t}</Button>)}
              </div>}
          </div>)}
          </div>
        </>:<Text>当前未检测到超过阈值的应用声音。播放声音后会自动刷新，也可点击立即扫描。</Text>}
        <Text>标签由本地进程信息初筛为「待确认」，「为她所控」或 AI 可结合窗口内容复核。标签更新后，不会重复推送相同状态。</Text>
        <Text>AI 主程序及子进程固定排除于其他应用的独立采集；Windows 仍正常播放其声音。{appCapture.ai_host_pinned?"已固定主程序路径。":"如果主程序名称不同，可在上方有声列表固定一次。"}</Text>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"8px"}}>
          <Text>允许其他插件读取当前应用的结构化声音摘要</Text>
          <Switch checked={appCapture.external_channel_share===true}
            onChange={(value:boolean)=>setDialog({kind:"channel-share",enabled:value})}/>
        </div>
        <Text>仅输出候选歌词、视频语音文字和伴奏事件；不共享原始音轨、不自动发聊天消息。</Text>
        <Text>逐应用声道不能拆开同一浏览器进程内的多个标签页。系统混合回环无法保证排除 AI 声音。</Text>
      </Stack></Card>
      <Card title="音频类别 · 本地判定"><Stack>
        <StatusBadge tone={scene.kind==="mixed"||scene.kind==="ai_output"?"warning":
          sceneVerified?"success":"neutral"}>{sceneName}</StatusBadge>
        {appCapture.mode==="application"?<Text>
          {"来源："+(selectedApp.name?short(selectedApp.name,"所选应用",40)+" · PID "+String(selectedApp.pid||0):"尚未选择应用")}
          {selectedApp.name?(selectedApp.tag_status==="external_review"
            ?" · 复核："+(selectedApp.tag==="video"?"视频":selectedApp.tag==="music"?"音乐":selectedApp.tag==="other"?"其他":"未分类")
              +"（"+short(selectedApp.tag_origin,"外部复核",24)+"）"
            :" · 标签仍是候选，未通过复核"):""}
        </Text>:null}
        <Text>{sceneReasonNames[String(scene.reason||"")]||short(scene.reason,"尚无判定",60)}</Text>
        <Text>{appCapture.mode==="application"?
          (scene.kind==="video"
            ? (sceneVerified?"已有人声证据：视频语音转写只在这样的窗口工作"
               :"视频类别已确认，但这一窗没有人声，暂不启动转写")
            : "当前声道按 PID 隔离；内容类型由复核标签与本地声学信号共同判定，不能隔离同一进程的多个标签页")
          : (sceneVerified?"播放器报告了媒体类型；其他应用仍可能有声音"
             :"安全模式：无法验证单一来源时暂停识别和 AI 共享")}</Text>
        <Text>{appCapture.mode==="application"?
          "人声评分 "+displayScore(scene.speech_score)+" · 音乐评分 "+displayScore(scene.music_score)+"（YAMNet 本地证据，不是内容结论）"
          :"检测到 "+String(Number(scene.source_count||0))+" 个媒体会话来源"}</Text>
      </Stack></Card>
      <Card title="输出设备 · 高级设置"><Stack>
        <Text>日常使用无需单独切换「AI 正在说话」。选择独立应用声道后会排除固定的 AI 主程序；混合回环不能逐进程扣除声音。</Text>
        <Button disabled={Boolean(busy)} onClick={()=>setAdvancedOutputs(!advancedOutputs)}>{advancedOutputs?"收起设备设置":"展开扬声器设置"}</Button>
        {advancedOutputs?<>
          <Button disabled={Boolean(busy)} onClick={()=>{
            if(busy)return
            setBusy("list_outputs");setOutputsError("")
            void props.api.call("list_audio_outputs",{}, {userInitiated:true})
              .then(unpack).then((data:any)=>{
                if(data.ok===false)throw Error(String(data.error||"设备枚举失败"))
                setAudioOutputs(Array.isArray(data.devices)?data.devices:[])
              }).catch((e:any)=>setOutputsError(short(e?.message||e)))
              .finally(()=>setBusy("") )
          }}>查看输出设备</Button>
          {outputsError?<Alert tone="warning">{outputsError}</Alert>:null}
          <Text>监听设备：{status.capture_output_id?"已指定输出":"系统默认扬声器"}</Text>
          {audioOutputs.map((out:any,i:number)=><div key={String(out.id)+i}
            style={{display:"flex",gap:"8px",alignItems:"center",justifyContent:"space-between"}}>
            <Text>{short(out.name)}{out.is_default?" · 系统默认":""}</Text>
            <Button disabled={Boolean(busy)||listening} onClick={()=>setDialog({kind:"capture-output",output_id:out.id,output_name:out.name,ai_separate:false})}>监听此设备</Button>
          </div>)}
          {audioOutputs.length?<Button disabled={Boolean(busy)||listening}
            onClick={()=>setDialog({kind:"capture-output",output_id:"",output_name:"系统默认",ai_separate:false})}>恢复默认扬声器</Button>:null}
          <Text>系统混音不具备 AI 声音物理排除能力；需要绝对隔离请保持单应用进程采集。</Text>
        </>:null}
      </Stack></Card>
    </>:null}

    {tab==="video"?(status.suokong_available===false?
      <div style={{position:"relative"}}>
        <div style={{filter:"blur(3px)",opacity:0.45,pointerEvents:"none",userSelect:"none"}} aria-hidden="true">
          <Card title="视频语音状态"><Stack>
            <StatusBadge tone="neutral">暂无视频语音片段</StatusBadge>
            <Text>视频识别片段 0　·　疑似幻觉已排除 0</Text>
            <Text>检测到 0 个媒体会话来源</Text>
          </Stack></Card>
          <Card title="观看模式"><Stack>
            <SegmentedControl value="balanced"
              options={[{value:"balanced",label:"画面+语音（默认）"},{value:"efficient",label:"效能（仅画面）"}]}
              onChange={()=>{}}/>
            <Text>当前：画面+语音，抽帧看画面，语音作为辅助证据按时间对齐</Text>
          </Stack></Card>
        </div>
        <div style={{position:"absolute",inset:0,borderRadius:"18px",
          background:"#0b1622aa",display:"flex",alignItems:"center",justifyContent:"center",
          flexDirection:"column",gap:"10px",textAlign:"center",padding:"24px"}}>
          <div style={{fontSize:"26px",color:"#8ef6f0"}}>◎</div>
          <Text>请安装「为她所控」后可使用</Text>
          <Text>视频识别需要它来截取画面、匹配歌词和反查视频信息。</Text>
        </div>
      </div>
    :<>
      <Card title="视频语音状态"><Stack>
        <StatusBadge tone={Number(status.video_speech_segments||0)>0?"success":"neutral"}>
          {Number(status.video_speech_segments||0)>0?"已识别视频语音":"暂无视频语音片段"}</StatusBadge>
        <Text>视频识别片段 {Number(status.video_speech_segments||0)}　·　疑似幻觉已排除 {Number(status.video_suspect_cues||0)}</Text>
        {status.video_last_error?<Alert tone="warning">最近一次视频语音失败：{short(String(status.video_last_error),"",160)}</Alert>:null}
        <Text>检测到 {Number(scene.source_count||0)} 个媒体会话来源</Text>
        {mediaSources.length?mediaSources.map((src:any,i:number)=><div key={String(src.app)+":"+i}
          style={{padding:"9px 0",borderBottom:"1px solid #80808022"}}>
          <Text>{short(src.app,"未知应用")} · {src.playback_type==="video"?"报告视频":src.playback_type==="music"?"报告音乐":"类型未上报"}</Text>
          <Text>播放器提供的媒体类别仅作为候选；窗口复核后可由 AI 修正应用标签。</Text>
        </div>):<Text>当前播放器没有报告可识别的媒体会话；无法可靠判断"视频"。</Text>}
      </Stack></Card>
      <Card title="观看模式"><Stack>
        <SegmentedControl value={String(status.video_watch_mode||"balanced")}
          options={[{value:"balanced",label:"画面+语音（默认）"},{value:"efficient",label:"效能（仅画面）"}]}
          onChange={(mode:string)=>perform("set_video_mode",{mode},"已保存观看模式")}/>
        <Text>{String(status.video_watch_mode||"balanced")==="efficient"
          ?"当前：效能模式，只按抽帧看画面，不听语音，适合低配电脑"
          :"当前：画面+语音，抽帧看画面，语音作为辅助证据按时间对齐"}</Text>
      </Stack></Card>
    </>):null}

    {tab==="model"?<>
      <Card title="模型管理工作台"><Stack>
        <div style={{padding:"17px",borderRadius:"17px",background:"linear-gradient(120deg,#242d55,#22586e)",
             color:"#f5f7ff",display:"grid",gap:"11px"}}>
          <div style={{display:"flex",gap:"12px",alignItems:"center",flexWrap:"wrap"}}>
            <div style={{fontSize:"23px"}}>⬡</div>
            <div style={{flex:1,minWidth:0}}>
              <div style={{fontSize:"18px",fontWeight:850}}>模型中心</div>
              <div style={{fontSize:"12px",opacity:.78,marginTop:"4px"}}>3 套模型分开管理 · GPU 显存按需分配</div>
            </div>
            <div style={{padding:"7px 10px",fontSize:"12px",background:"#ffffff18",borderRadius:"22px"}}>
              {transferJobs.some(job=>transferActive(job.info))?"◉ 下载任务运行中":"✓ 无下载任务"}</div>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,minmax(0,1fr))",gap:"8px"}}>
            {[{name:"专业歌词",value:"中 / 日 / 英"},{name:"视频语音",value:"Whisper"},
              {name:"分离模型",value:"四档音质"}].map((v:any,i:number)=><div key={i}
              style={{padding:"10px",borderRadius:"12px",background:"#ffffff12"}}>
              <div style={{fontSize:"10px",opacity:.72}}>{v.name}</div>
              <div style={{fontSize:"13px",fontWeight:750,marginTop:"6px"}}>{v.value}</div>
            </div>)}
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"8px",flexWrap:"wrap"}}>
          <div style={{fontWeight:800,fontSize:"15px"}}>下载任务</div>
          <span style={{fontSize:"12px",opacity:.65}}>{transferJobs.length?transferJobs.length+" 个任务":"待机 · 下载前需要确认"}</span>
        </div>
        {transferJobs.length?transferJobs.map(job=><DownloadProgress key={job.key} title={job.title}
          task={job.info} size={job.size} accent={job.accent}
          onRetry={job.onRetry} /> ):
          <div style={{padding:"18px",borderRadius:"13px",border:"1px dashed #819ab34f",
            background:"rgba(86,129,169,.055)",display:"flex",gap:"13px",alignItems:"center"}}>
            <div style={{fontSize:"24px",opacity:.7}}>◷</div>
            <div><div style={{fontWeight:750,fontSize:"13px"}}>还没有正在传输的模型</div>
              <div style={{fontSize:"12px",opacity:.7,marginTop:"3px"}}>选择下方模型 → 确认下载 → 此处显示速度、进度和时间。</div>
            </div>
          </div>}
        <div style={{display:"flex",gap:"9px",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap"}}>
          <div style={{fontSize:"12px",opacity:.8}}>下载来源 · 模型权重均需完整校验</div>
          <SegmentedControl value={source} options={sources} onChange={(v:string)=>setDialog({kind:"source",value:v})}/>
        </div>
      </Stack></Card>
      <Card title="显存 · 自适应调度"><Stack>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"12px",flexWrap:"wrap"}}>
          <div>
            <Text><strong>按应用标签按需运行 · 音乐 / 视频互斥加载</strong></Text>
            <Text>已确认音乐启动歌声，已确认视频启动语音；混音 / 待确认不加载识别模型。</Text>
          </div>
          <StatusBadge tone={resources.pressure_detected?"warning":"neutral"}>
            {gpu.available===true?`GPU 可用 ${gpu.free_mib} / ${gpu.total_mib} MiB`:
              "GPU 可用显存尚未测得"}
          </StatusBadge>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(2,minmax(0,1fr))",gap:"10px"}}>
          <div style={{borderRadius:"13px",padding:"14px",background:"rgba(55,148,167,.075)",border:"1px solid #3195aa33"}}>
            <div style={{fontSize:"11px",opacity:.68}}>可用显存</div>
            <div style={{fontSize:"21px",fontWeight:850,marginTop:"5px"}}>
              {gpu.available===true?mb(Number(gpu.free_mib)*1048576):"尚未测得"}</div>
          </div>
          <div style={{borderRadius:"13px",padding:"14px",background:"rgba(85,126,224,.075)",border:"1px solid #4f87de33"}}>
            <div style={{fontSize:"11px",opacity:.68}}>运行策略</div>
            <div style={{fontSize:"15px",fontWeight:800,marginTop:"7px"}}>
              {resources.enabled!==false?"自适应降档":"保持手动档位"}</div>
          </div>
        </div>
        <Text>只使用已下载的低档模型；CPU 内存回退需明确授权。</Text>
        <div style={{display:"flex",justifyContent:"space-between",gap:"8px",alignItems:"center"}}>
          <Text>自动降低显存占用</Text>
          <Switch checked={resources.enabled!==false} onChange={(value:boolean)=>
            requestResourcePolicy("video","enabled",{enabled:value,reserve_mib:reserveMiB,
              cpu_fallback:Boolean(cpuPermissions.video)})}/>
        </div>
        <Text>为游戏和其他程序预留显存</Text>
        <SegmentedControl value={String(reserveMiB)} options={[
          {value:"1024",label:"1 GB"},{value:"2048",label:"2 GB"},{value:"4096",label:"4 GB"}]}
          onChange={(value:string)=>requestResourcePolicy("video","reserve",{
            enabled:resources.enabled!==false,reserve_mib:Number(value),
            cpu_fallback:Boolean(cpuPermissions.video)})}/>
        {resources.pressure_detected?<Alert tone="warning">
          检测到持续显存紧张；下一段音频将按已下载模型降档。当前音频处理不强制中断。
        </Alert>:null}
      </Stack></Card>
      <Card title="专业歌声歌词 · 中日英"><Stack>
        <ModelSpotlight kind="SINGING RECOGNITION" icon="♫" title="专业歌声歌词" 
          subtitle="只处理音乐人声 · 时间轴只供 AI 读取"
          model={singingOptions.find(q=>q.value===singingQuality)?.name||"待选择"}
          selected={tierLabels[singing.active_quality]||"未启用"}
          device={singing.runtime_device==="cuda"?(resourceModels.singing?.placement==="gpu_ram_weights"?"GPU + 内存权重分层":"NVIDIA GPU"):singing.runtime_device==="cpu"?"CPU 内存":"尚未验证"}
          accent="#8e83ea"/>
        <SegmentedControl value={singingQuality} options={singingOptions.map(q=>({value:q.value,label:q.label}))}
          onChange={(q:string)=>{
            setSingingQuality(q);setError("");setNotice("")
            const record=(singing.models||{})[q]||{}
            if (record.state!=="verified" && record.state!=="running")
              setDialog({kind:"singing-download",quality:q})
          }}/>
        <div style={{padding:"10px 12px",background:"rgba(141,125,233,.07)",borderRadius:"12px"}}>
          <Text>已选：{modelStates[(singing.models||{})[singingQuality]?.state]||"未安装"}　·　自适应实际档：{tierLabels[resourceModels.singing?.quality]||"待选"}</Text>
        </div>
        <div style={{display:"flex",justifyContent:"space-between",gap:"12px",alignItems:"center"}}>
          <Text>Qwen 显存不足：尝试 GPU + 系统内存权重分层（实验）</Text>
          <Switch checked={resources.gpu_ram_offload_enabled!==false} onChange={(value:boolean)=>
            requestResourcePolicy("singing","gpu_ram_offload",{enabled:resources.enabled!==false,
              reserve_mib:reserveMiB,cpu_fallback:Boolean(cpuPermissions.singing),gpu_ram_offload:value})}/>
        </div>
        <Text>显卡仍负责部分计算，部分权重和计算会占用内存与 CPU；可能明显变慢。只支持 Qwen 歌声模型；Windows 驱动的共享显存不能强制分配。</Text>
        <Text>运行位置 · 系统内存 / NVIDIA 显存</Text>
        <SegmentedControl value={singingDevice} options={devices} onChange={(d:string)=>setDialog({kind:"singing-device",value:d})}/>
        <div style={{display:"flex",justifyContent:"space-between",gap:"10px",alignItems:"center"}}>
          <Text>显存不足时允许改用系统内存（CPU 速度可能很慢）</Text>
          <Switch checked={Boolean(cpuPermissions.singing)} onChange={(value:boolean)=>
            requestResourcePolicy("singing","cpu_fallback",{enabled:resources.enabled!==false,
              reserve_mib:reserveMiB,cpu_fallback:value})}/>
        </div>
        {resourceModels.singing?.requires_cpu_consent?<Alert tone="warning">已安装模型无法满足当前显存预算。请确认 CPU 内存运行，或先安装较低档模型。</Alert>:null}
        <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
          <Button disabled={Boolean(busy)||singingRemoving||singing.preparation?.state==="running"||singing.activation?.state==="running"}
            onClick={()=>setDialog({kind:"singing-download",quality:singingQuality})}>下载 / 校验</Button>
          <Button disabled={Boolean(busy)||singingRemoving||singing.preparation?.state==="running"}
            onClick={()=>setDialog({kind:"singing-import",quality:singingQuality})}>核验手动导入</Button>
          <Button disabled={Boolean(busy)||singingRemoving||singing.activation?.state==="running"||((singing.models||{})[singingQuality]||{}).state!=="verified"}
            onClick={()=>setDialog({kind:"singing-activate",quality:singingQuality})}>验证并启用</Button>
          <Button disabled={Boolean(busy)||singingInstalling||singingRemoving||
                  singing.activation?.state==="running"||
                  ((singing.models||{})[singingQuality]||{}).state==="missing"}
            onClick={()=>setDialog({kind:"singing-uninstall",quality:singingQuality})}>卸载此档模型</Button>
        </div>
        {singing.activation?.state==="running"||singing.activation?.state==="failed"?
          <Text>验证：{short(singing.activation?.stage)} · {short(singing.activation?.error||"处理中")}</Text>:null}
        {(singing.preparation?.state==="running"||singing.preparation?.state==="failed") &&
           singing.preparation?.quality===singingQuality?<Text>
             {singing.preparation?.state==="running"?"模型正在后台准备；速度、百分比和校验阶段见上方「下载任务」。":
              "本次任务未完成；原因和重试按钮见上方「下载任务」，无需重新下载已验证的模型。"}
           </Text>:null}
        <ModelTiles tiers={singingOptions} states={singing.models||{}}
          chosen={singingQuality} accent="#8e83ea"/>
        <div style={{borderRadius:"16px",border:"1px solid rgba(129,118,214,.28)",
            background:"linear-gradient(125deg,rgba(146,128,223,.09),rgba(88,161,192,.045))",
            padding:"16px",display:"grid",gap:"11px"}}>
          <div style={{display:"flex",alignItems:"center",gap:"11px",justifyContent:"space-between",flexWrap:"wrap"}}>
            <div style={{display:"flex",alignItems:"center",gap:"10px"}}>
              <div style={{fontSize:"20px",background:"#8a82ee20",borderRadius:"12px",padding:"8px"}}>▣</div>
              <div><div style={{fontWeight:800,fontSize:"14px"}}>专业歌声 · 独立运行环境</div>
                <div style={{fontSize:"12px",opacity:.76,marginTop:"3px"}}>
                  {singingRemoving?"正在卸载环境":singingEnvironment.python_present?"独立 Python 已创建 · 依赖需核验":"独立 Python 尚未安装或不完整"}
                </div>
              </div>
            </div>
            <StatusBadge tone={singingRemoving?"warning":singingEnvironment.python_present?"success":"neutral"}>
              {singingRemoving?"卸载中":singingEnvironment.python_present?"已创建 · 未验证":"待安装"}
            </StatusBadge>
          </div>
          <div style={{display:"flex",gap:"9px",flexWrap:"wrap"}}>
            {singingInstalling?<Button disabled={Boolean(busy)||singingRemoving}
                onClick={()=>setDialog({kind:"singing-install-cancel"})}>停止当前安装</Button>:null}
            <Button disabled={Boolean(busy)||singingRemoving}
              onClick={()=>setDialog({kind:"singing-environment-uninstall"})}>
              卸载独立环境</Button>
          </div>
          <div style={{fontSize:"12px",opacity:.8}}>环境卸载只删除 singing_asr_venv，不删除已下载歌词权重、视频模型、分离模型或公共 CUDA。</div>
          {singing.preparation?.state==="failed"?<Text>
            本次安装失败。完整原因见上方下载任务；安装日志保留，方便区分依赖安装与模型下载故障。
          </Text>:null}
          {environmentRemoval.state==="failed"?<Alert tone="warning">
            卸载失败：{short(environmentRemoval.error,"独立环境仍被占用",380)}
          </Alert>:null}
          {environmentRemoval.state==="ready"?<Text>{short(environmentRemoval.note,"独立环境卸载完成",200)}</Text>:null}
          {singing.preparation?.state==="interrupted"?<Text>安装已取消；可以安全卸载环境或再次确认安装。</Text>:null}
          <div style={{fontSize:"11px",opacity:.7,overflowWrap:"anywhere"}}>
            安装诊断日志：{short(singingEnvironment.install_log,"插件 data/environment/singing_asr_install.log",320)}
          </div>
        </div>
        <Text>极高档错误率 ≤5% 是待验收目标，无法保证每首歌达到。</Text>
      </Stack></Card>
      <Card title="视频人物说话 · Whisper（原模型保留）"><Stack>
        <ModelSpotlight kind="VIDEO SPEECH" icon="▶" title="视频人物说话"
          subtitle="处理视频语音 · 自动避开正在使用的歌声模型"
          model={"Whisper "+model.name} selected={tierLabels[activeQuality]||"未启用"}
          device={status.lyrics_runtime_device==="cuda"?"NVIDIA GPU":
            status.lyrics_runtime_device==="cpu"?"CPU 内存":"尚未验证"} accent="#4b9af4"/>
        <SegmentedControl value={quality} options={qualities.map(q=>({value:q.value,label:q.label}))}
          onChange={(q:string)=>{
            setQuality(q);setHasPicked(true);setError("");setNotice("")
            const selected=qualities.find(x=>x.value===q)
            const record=(catalog.models||{})[selected?.model||""]||{}
            if (record.state!=="verified" && record.state!=="downloading" && record.state!=="verifying")
              setDialog({kind:"lyrics-download",quality:q})
          }}/>
        <div style={{padding:"10px 12px",background:"rgba(75,154,244,.07)",borderRadius:"12px"}}>
          <Text>已选：{modelStates[item.state]||"未安装"}　·　自适应实际档：{tierLabels[resourceModels.video?.quality]||"待选"}</Text>
        </div>
        <div style={{display:"flex",justifyContent:"space-between",gap:"12px",alignItems:"center"}}>
          <Text>Qwen 显存不足：尝试 GPU + 系统内存权重分层（实验）</Text>
          <Switch checked={resources.gpu_ram_offload_enabled!==false} onChange={(value:boolean)=>
            requestResourcePolicy("singing","gpu_ram_offload",{enabled:resources.enabled!==false,
              reserve_mib:reserveMiB,cpu_fallback:Boolean(cpuPermissions.singing),gpu_ram_offload:value})}/>
        </div>
        <Text>显卡仍负责部分计算，部分权重和计算会占用内存与 CPU；可能明显变慢。只支持 Qwen 歌声模型；Windows 驱动的共享显存不能强制分配。</Text>
        <Text>运行位置 · 系统内存 / NVIDIA 显存</Text>
        <SegmentedControl value={device} options={devices} onChange={(d:string)=>setDialog({kind:"device",value:d})}/>
        <div style={{display:"flex",justifyContent:"space-between",gap:"10px",alignItems:"center"}}>
          <Text>显存不足时允许改用系统内存（CPU 速度可能很慢）</Text>
          <Switch checked={Boolean(cpuPermissions.video)} onChange={(value:boolean)=>
            requestResourcePolicy("video","cpu_fallback",{enabled:resources.enabled!==false,
              reserve_mib:reserveMiB,cpu_fallback:value})}/>
        </div>
        {resourceModels.video?.requires_cpu_consent?<Alert tone="warning">当前显存无法容纳已安装的候选档位。可以降档或授权 CPU 内存运行。</Alert>:null}
        {loading?<DownloadProgress title={"视频语音 · Whisper "+model.name} task={item}
          size={item.plan?.download_bytes_estimate} accent="#4b9af4"/>:null}
        {failed?<Alert tone="warning">{short(item.message||"下载未完成，支持继续")}</Alert>:null}
        <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
          <Button disabled={Boolean(busy)||loading||paused}
            onClick={()=>setDialog({kind:"lyrics-download",quality})}>{verified?"重新校验":(failed?"继续下载":"下载 / 校验")}</Button>
          <Button disabled={Boolean(busy)||!verified||paused||switchBusy}
            onClick={()=>setDialog({kind:"lyrics-activate",quality})}>验证并启用</Button>
          <Button disabled={Boolean(busy)||item.state==="missing"||loading}
            onClick={()=>setDialog({kind:"lyrics-uninstall",quality})}>卸载</Button>
          <Button disabled={Boolean(busy)||paused||switchBusy}
            onClick={()=>perform("fallback_verified_lyrics_model",{},"尝试恢复可用视频语音模型")}>恢复旧模型</Button>
        </div>
        {switchBusy||switchState.status==="failed"?<Text>{short(switchState.message||"正在模型验证")}</Text>:null}
        <ModelTiles tiers={qualities} states={Object.fromEntries(
          qualities.map(q=>[q.value, (catalog.models||{})[q.model]||{}]))}
          chosen={quality} accent="#4b9af4"/>
        <Text>视频模型不作为音乐歌词识别器；切换音源后自动卸载不活跃的识别模型。</Text>
      </Stack></Card>
    </>:null}

    {tab==="diagnose"?<Card title="自动运行检测"><Stack>
      <StatusBadge tone={health.state==="failed"?"warning":health.state==="complete"?"success":"neutral"}>
        {health.state==="running"?"启动后正在自动检查":
         health.state==="failed"?"自动检测发现异常":
         health.state==="complete"?"自动检测完成":health.state==="limited"?"基础检查完成 · 重型推理延后":"准备自动检查"}
      </StatusBadge>
      <Text>{short(health.stage,"启动后自动检测环境与有声应用，无需手动运行")}</Text>
      {health.state==="limited"?<Alert tone="warning">{short(health.message,"当前有任务占用，深度测试暂缓")}</Alert>:null}
      {health.state==="failed"?<Alert tone="warning">{short(health.error,"环境异常，需人工确认")}</Alert>:null}
      <Text>Python 环境：{env.ready?"可用":"未就绪"} · GPU 实际：
        {status.lyrics_runtime_device==="cuda"?"CUDA 已使用":status.lyrics_runtime_device==="cpu"?"CPU":"待实际推理确认"}</Text>
      <Text>歌词队列：{Number(lyrics.queue_depth||0)} 段 · 前后台版本：{mismatchedBuild?"不一致":"一致"}</Text>
      <div style={{display:"flex",gap:"9px",flexWrap:"wrap"}}>
        <Button disabled={Boolean(busy)} onClick={()=>{
          if(busy)return
          setBusy("log");setError("")
          void props.api.call("get_lyrics_diagnostics",{}, {userInitiated:true})
            .then(unpack).then((result:any)=>{setDiagnostics(result);setShowLog(true)})
            .catch((e:any)=>setError(short(e?.message||e))).finally(()=>setBusy(""))
        }}>{showLog?"刷新诊断详情":"诊断详情"}</Button>
        <Button disabled={Boolean(busy)}
          onClick={()=>perform("automatic_audio_check",{},"已重新开始检测")}>重新检查</Button>
      </div>
      {showLog?<>
        <Button onClick={()=>setShowLog(false)}>收起详情</Button>
        <pre style={{whiteSpace:"pre-wrap",overflowWrap:"anywhere",maxHeight:"260px",overflow:"auto",fontSize:"12px"}}>
          {JSON.stringify(diagnostics,null,2).slice(0,4500)}
        </pre>
      </>:null}
      <Text>启动时自动检查声卡会话、隔离依赖和已安装模型文件。只有音源经 AI / 窗口确认，才允许按需执行高显存推理。不会未经确认下载新权重。</Text>
    </Stack></Card>:null}

    {tab==="groove"?<Card title="律动"><Stack>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"12px"}}>
        <div><Text>角色跟着音乐轻轻摆动</Text>
          <Text>关掉后只是不摆，识别和歌词都照常。</Text></div>
        <Switch checked={groove.enabled!==false} disabled={Boolean(busy)}
          onChange={(on:boolean)=>{perform("set_live2d_groove",{enabled:on},on?"律动已开启":"律动已关闭")}}/>
      </div>
      <StatusBadge tone={groove.enabled===false?"neutral":
        (groove.needs_restart?"warning":
         (grooveLive.listening?"success":grooveBridge.running?"success":"neutral"))}>
        {groove.enabled===false?"已关闭":
         groove.needs_restart?"重启后生效":
         grooveLive.listening?"正在律动":"等待音乐"}
      </StatusBadge>
      <Text>{short(grooveLive.song,"还没听到歌",60)}</Text>
      <div style={{display:"flex",gap:"10px",alignItems:"center"}}>
        {[{label:"能量",value:grooveLive.energy},{label:"低",value:grooveLive.low},
          {label:"中",value:grooveLive.mid},{label:"高",value:grooveLive.high}].map((m:any,i:number)=>
          <div key={i} style={{flex:1,minWidth:0}}>
            <div style={{height:"22px",display:"flex",alignItems:"flex-end"}}>
              <div style={{width:"100%",height:Math.max(2,groovePct(m.value)*0.22)+"px",
                background:"#2db6a4",borderRadius:"3px"}}/>
            </div>
            <div style={{fontSize:"10px",color:"#adcee1",marginTop:"4px"}}>{m.label}</div>
          </div>)}
      </div>
      {groove.needs_restart?<Text>重启 N.E.K.O 后生效（不用刷新页面）。</Text>:null}
      <Text>只用已有频谱，不额外推理、不上传。</Text>
    </Stack></Card>:null}
    {tab==="settings"?<Card title="分离与人声增强"><Stack>
      <Button onClick={()=>{perform("onboarding_reopen",{},"已打开新手引导");setGuideStep(1);setShowGuide(true)}}>
        重新运行新手引导</Button>
      <Divider/>
      <ModelSpotlight kind="VOCAL SEPARATION" icon="♬" title="人声与伴奏"
        subtitle="按需分离 · 试听原声和增强结果"
        model={stemChoices.find(q=>q.value===stemQuality)?.name||"MDX"}
        selected={tierLabels[stem.quality]||"待选择"}
        device={stem.actual_device==="cuda"?"NVIDIA GPU":stem.actual_device==="cpu"?"CPU 内存":"尚未验证"}
        accent="#2db6a4"/>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"12px"}}>
        <div><Text>人声 / 伴奏分离</Text>
          <Text>{stem.enabled?"已经开启 · 按需分析":"目前关闭 · 不额外消耗模型资源"}</Text></div>
        <Switch checked={Boolean(stem.enabled)} disabled={Boolean(busy)} onChange={(on:boolean)=>{
          if(!on) perform("set_stem_separation",{enabled:false},"分离已关闭")
          else if(!stemVerified)setDialog({kind:"stem-download",quality:stem.quality||stemQuality})
          else setDialog({kind:"stem-enable"})
        }}/>
      </div>
      <Text>四档不同的本地分离模型（性能与音质不保证随档位单调提升，建议逐档试听）</Text>
      <SegmentedControl value={stemQuality} options={stemChoices.map(x=>({value:x.value,label:x.label}))}
        onChange={(v:string)=>setStemQuality(v)}/>
      <ModelTiles tiers={stemChoices} states={stemModels} chosen={stemQuality} accent="#2db6a4"/>
      <div style={{padding:"14px",border:"1px solid #2db6a442",borderRadius:"15px",
           background:"rgba(45,182,164,.055)",display:"grid",gap:"10px"}}>
        <Text><strong>{stemChoices.find(x=>x.value===stemQuality)?.name||"分离模型"}</strong> ·
          {stemModels[stemQuality]?.present?" 本地已发现文件":" 尚未安装"}</Text>
        <Text>{short(stemModels[stemQuality]?.note,"需要自行试听对比实际音质")}</Text>
        <Text>模型文件约 {mb(stemModels[stemQuality]?.expected_bytes_estimate||0)}；
          建议可用显存 {Number(stemModels[stemQuality]?.vram_estimate_mib||0)} MiB（估计）</Text>
        <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
          <Button disabled={Boolean(busy)||stemProgressActive||paused}
            onClick={()=>setDialog({kind:"stem-download",quality:stemQuality})}>
            {stemModels[stemQuality]?.present?"下载 / 核验":"下载安装"}</Button>
          <Button disabled={Boolean(busy)||stemProgressActive||!stemModels[stemQuality]?.present}
            onClick={()=>setDialog({kind:"stem-select",quality:stemQuality})}>验证并启用</Button>
          <Button disabled={Boolean(busy)||stemProgressActive||!stemModels[stemQuality]?.present}
            onClick={()=>setDialog({kind:"stem-uninstall",quality:stemQuality})}>卸载此档</Button>
        </div>
      </div>
      <Text>生效档位：{tierLabels[stem.quality]||"极高"} · 运行设备：{stem.actual_device==="cuda"?(gpuProbe.real_inference===true?"NVIDIA GPU · 已实测":"CUDA Provider · 待实测"):stem.actual_device==="cpu"?"CPU":"尚未验证"}</Text>
      {stem.device_preference==="cuda"&&stem.actual_device==="cpu"?<Alert tone="warning">
        当前 CPU 兜底：{short(stem.gpu_fallback_reason||"CUDA Provider 尚未验证，请安装 GPU 加速并检测","",210)}
      </Alert>:null}
      <div style={{border:"1px solid #80808030",borderRadius:"12px",padding:"12px",display:"grid",gap:"8px"}}>
        <Text><strong>插件专用 GPU 环境</strong> · {stemGpu.installed?"ONNX + CUDA/cuDNN 已安装":"CUDA/cuDNN 尚未完整安装"}</Text>
        <Text>安装状态：{gpuInstall.state==="running"?"正在准备":gpuInstall.state==="ready"?"安装完成":gpuInstall.state==="failed"?"安装失败":gpuInstall.state==="interrupted"?"准备中断":"等待操作"} · 实际 CUDA：{gpuProbe.real_inference===true?"试推理通过":gpuProbe.cuda_provider?"Provider 已找到，仍需试推理":"尚未验证"}</Text>
        {gpuInstall.state==="running"?<DownloadProgress title={"GPU 环境 · "+short(gpuInstall.package||"ONNX + CUDA")}
          task={gpuInstall} size={stemGpu.download_bytes_estimate} accent="#f6a065"/>:null}
        {["failed","interrupted"].includes(gpuInstall.state)?<Alert tone="warning">{short(gpuInstall.message||"GPU 安装未完成","",220)}</Alert>:null}
        {gpuProbe.error?<Alert tone="warning">{short(gpuProbe.error,"",240)}</Alert>:null}
        {gpuProbe.native_loader_hint?<Text>{short(gpuProbe.native_loader_hint,"",300)}</Text>:null}
        {gpuProbe.suggestion?<Text>{short(gpuProbe.suggestion,"",260)}</Text>:null}
        <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
          <Button disabled={Boolean(busy)||stemGpu.running} onClick={()=>setDialog({kind:"stem-gpu-install"})}>
            {stemGpu.installed?"重新检查安装":"安装 GPU 分离加速"}</Button>
          <Button disabled={Boolean(busy)||gpuProbe.state==="running"||stemGpu.running}
            onClick={()=>perform("check_stem_gpu",{},"已开始后台 GPU 兼容检测")}>GPU 实际推理检测</Button>
        </div>
        <Text>独立 Python + 私有 CUDA 12.x、cuDNN 9、ONNX GPU；Windows 只共用 NVIDIA 驱动和系统 VC++ 运行库。安装后仍须真实模型推理验证。</Text>
      </div>
      <Text>运行位置 · 系统内存 / NVIDIA 显存</Text>
      <SegmentedControl value={stem.device_preference||"cpu"} options={devices}
        onChange={(value:string)=>setDialog({kind:"stem-device",value})}/>
      <div style={{display:"flex",justifyContent:"space-between",gap:"10px",alignItems:"center"}}>
        <Text>显存不足时允许分离模型转到 CPU 内存（可能更慢）</Text>
        <Switch checked={Boolean(cpuPermissions.stem)} onChange={(value:boolean)=>
          requestResourcePolicy("stem","cpu_fallback",{enabled:resources.enabled!==false,
            reserve_mib:reserveMiB,cpu_fallback:value})}/>
      </div>
      <Text>分离生效档位：{tierLabels[resourceModels.stem?.quality]||"待选择"} · {
        resourceModels.stem?.device==="cpu"?"系统内存":resourceModels.stem?.device==="cuda"?"NVIDIA 显存":"尚未验证"}</Text>
      {stemProgressActive && stemDownload.model===stemQuality?<DownloadProgress
        title="当前分离模型下载" task={stemDownload}
        size={stemModels[stemQuality]?.expected_bytes_estimate} accent="#2db6a4"/>:null}
      {(stemDownload.state==="failed"||stemDownload.state==="interrupted")&&stemDownload.model===stemQuality?
        <Alert tone="warning">{short(stemDownload.message||"文件未通过校验，可点击继续")}</Alert>:null}
      <Text>分离后人声二次处理 · 纯波形 / 频谱算法，不增加新模型</Text>
      <SegmentedControl value={stem.enhance_strength||"balanced"} options={enhanceModes}
        onChange={(strength:string)=>perform("set_stem_enhancement",{strength},"下一段应用新处理强度")}/>
      <Text>波形增强可能损坏咬字。下面可以试听并明确选定识别模型实际读取的音轨；默认原始分离人声。</Text>
      <Text>识别输入 · 不影响伴奏分析</Text>
      <SegmentedControl value={stem.vocal_input||"raw"}
        options={[{value:"raw",label:"原始人声 · 推荐"},{value:"enhanced",label:"增强人声"}]}
        onChange={(source:string)=>perform("set_stem_vocal_input",{source},"下一段歌词将使用所选人声音轨")}/>
      <Text>当前送给歌声模型：{stem.vocal_input==="enhanced"?"波形增强人声":"原始分离人声"}。两个版本均可通过下方试听按钮对比。</Text>
      <div style={{border:"1px solid #80808035",borderRadius:"15px",padding:"12px",display:"grid",gap:"10px"}}>
        <Text><strong>试听最近一段</strong> · 模型分离 / 自适应增强 / 伴奏</Text>
        <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
          {[{id:"raw",name:"① 原始分离人声"},{id:"enhanced",name:"② 波形增强人声"},
            {id:"instrumental",name:"伴奏"}].map(x=><Button key={x.id}
              disabled={Boolean(busy)||Boolean(previewBusy)||!stem.preview_available}
              onClick={()=>previewSound(x.id)}>{previewBusy===x.id?"正在准备":x.name}</Button>)}
        </div>
        {preview?.uri?<>
          <Text>试听轨道：{preview.stage==="raw"?"模型分离人声":preview.stage==="enhanced"?"增强后人声":"伴奏"} ·
          约 {Number(preview.duration||0).toFixed(1)} 秒</Text>
          <audio controls preload="none" src={preview.uri} style={{width:"100%"}}/>
        </>:<Text>开启监听并播放音乐后，会保存最近7秒的短片段供你主动试听（不写磁盘）。</Text>}
        <Text>试听为 Whisper 使用的 16kHz 单声道版本；若使用系统混音，播放试听可能再次被监听，建议暂停插件后试听。</Text>
      </div>
      {stem.error?<Alert tone="warning">{short(stem.error)}</Alert>:null}
      <Text>歌词语种</Text>
      <SegmentedControl value={language} options={languages} onChange={(value:string)=>{
        setLanguage(value);perform("set_lyrics_language",{language:value},"已保存语言设置")
      }}/>
      <Text>AI 只收到用户允许的结构化分析与候选文字，不会因为试听按钮自动收到音频。</Text>
    </Stack></Card>:null}

  </Page>
  {dialog?<Modal open={true} title={dialogTitle} size="md"
      closeOnBackdrop={false} closeOnEscape={true} lockScroll={false}
      onClose={()=>setDialog(null)}
      footer={        <div style={{display:"flex",justifyContent:"flex-end",gap:"9px",padding:"12px 16px",
           borderTop:"1px solid #dbe6f5",background:"#f3f7fd",flexShrink:0,flexWrap:"wrap"}}>
          {dialog.kind==="health-error"?<Button onClick={()=>setDialog(null)}>明白</Button>:
          dialog.kind==="gpu-shortage"?<>
            <Button onClick={()=>setDialog(null)}>取消</Button>
            <Button onClick={()=>{const domain=dialog.domain;setDialog(null);applyDevice(domain,"cpu")}}>
              改用内存（CPU）</Button>
            <Button onClick={()=>{const domain=dialog.domain;setDialog(null);applyDevice(domain,"cuda")}}>
              仍尝试显存（GPU）</Button>
          </>:<>
            <Button onClick={()=>setDialog(null)}>取消</Button>
            <Button disabled={Boolean(busy)} onClick={onConfirm}>
              {dialog.kind?.includes("uninstall")?"确认安全卸载":
               dialog.kind==="app-install"||dialog.kind==="stem-gpu-install"?"确认安装":
               dialog.kind?.includes("download")?"确认并下载":
               dialog.kind==="stem-select"?"校验并启用":
               dialog.kind==="resource-policy"?"确认并保存授权":"确定"}
            </Button>
          </>}
        </div>
}>
      <div aria-label="音频控制台确认内容" style={{minWidth:0,color:"#253b5d"}}>
        <div onWheel={(e:any)=>e.stopPropagation()} style={{padding:"6px 3px 12px",overflowY:"auto",minHeight:0,maxHeight:"min(56dvh, 470px)",
           overscrollBehaviorY:"contain",touchAction:"pan-y",WebkitOverflowScrolling:"touch"}}>
        <Stack>
          {dialog.kind?.includes("download")?<div style={{borderRadius:"14px",padding:"15px",
            background:"linear-gradient(125deg,rgba(75,139,222,.16),rgba(69,162,163,.08))",
            border:"1px solid #79a0ce40",display:"grid",gap:"12px"}}>
            <div style={{fontSize:"14px",fontWeight:800}}>下载前确认资源</div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(3,minmax(0,1fr))",gap:"8px"}}>
              {[{name:"下载大小",value:mb(dialogIsSinging?singingPlans.download_bytes_estimate:
                dialog.kind==="stem-gpu-install"?stemGpu.download_bytes_estimate:dialogPlan.download_bytes_estimate)},
                {name:"建议磁盘",value:mb(dialogIsSinging?singingPlans.disk_required_bytes_estimate:
                   dialog.kind==="stem-gpu-install"?6000000000:dialogPlan.disk_required_bytes_estimate)},
                {name:"空闲显存建议",value:(dialogIsSinging?singingPlans.suggested_free_vram_mib:
                   dialogPlan.suggested_free_vram_mib)?String(dialogIsSinging?
                   singingPlans.suggested_free_vram_mib:dialogPlan.suggested_free_vram_mib)+" MiB":"按环境"}].map((metric:any,i:number)=><div key={i}
                style={{padding:"10px",borderRadius:"10px",background:"rgba(103,140,186,.1)",minWidth:0}}>
                  <div style={{fontSize:"10px",opacity:.72}}>{metric.name}</div>
                  <div style={{fontSize:"13px",fontWeight:800,marginTop:"6px",overflowWrap:"anywhere"}}>{metric.value}</div>
                </div>)}
            </div>
            <div style={{fontSize:"11px",opacity:.78}}>
              {source==="mirror"?"国内镜像优先":"官方优先"} · 下载过程会显示速度、进度与剩余时间 · SHA-256 校验后才能启用
            </div>
          </div>:null}
          {dialogIsLyrics?<Text>{modelDisplay.label} · {modelDisplay.name}</Text>:null}
          {dialogIsSinging?<div style={{display:"grid",gap:"11px"}}>
            <div style={{fontWeight:800,fontSize:"16px"}}>{tierLabels[dialog.quality]} · {singingName}</div>
            <div style={{fontSize:"13px",lineHeight:1.65}}>
              {dialog.kind==="singing-download"?"首次安装会准备私有 Python 依赖；确认后自动下载，并显示真实传输/校验状态。":
               dialog.kind==="singing-uninstall"?"仅卸载本档专业歌声模型。运行中会先释放模型进程；其他模型和环境保留。":
               dialog.kind==="singing-activate"?"此操作会先完成 SHA-256 校验，再实际加载模型和预热设备。":
               "仅检查已手动放入的本地模型文件，不会连接网络或删除文件。"}
            </div>
            {dialog.kind==="singing-import"?<div style={{fontSize:"12px",overflowWrap:"anywhere"}}>
              文件夹：data/models/singing_specialist/{(singing.models||{})[dialog.quality]?.id||"此档位"}/
            </div>:null}
            {dialog.kind==="singing-download"?<div style={{fontSize:"12px",opacity:.85}}>
              {source==="mirror"?"国内镜像优先":"官方优先"} · 私有依赖不写入系统 Python · 下载后必须完成校验
            </div>:null}
          </div>:null}
          {dialog.kind==="singing-environment-uninstall"?<>
            <div style={{fontSize:"15px",fontWeight:800}}>只删除歌声 Python 虚拟环境</div>
            <Text>不会卸载歌声模型权重、视频 Whisper、人声分离模型或 v0.9.5 私有 CUDA。</Text>
            <Alert tone="warning">会先停止本插件的专业歌声进程及正在执行的安装；下次使用需重新安装依赖。Windows 文件占用时明确报错。</Alert>
          </>:null}
          {dialog.kind==="singing-install-cancel"?<>
            <Text>停止当前专业歌声环境安装或模型下载，保留已下载文件和安装日志。</Text>
            <Text>取消完成后可以再次安装，或选择卸载独立环境。</Text>
          </>:null}
          {dialogIsStem?<Text>{short(stemChoices.find(x=>x.value===(dialog.quality||stemQuality))?.name,"MDX 分离模型")} · ONNX</Text>:null}
          {dialog.kind==="resource-policy"?<>
            <Text>模块：{dialog.domain==="singing"?"专业歌声":dialog.domain==="video"?"视频语音":"人声分离"}</Text>
            <Text>自适应：{dialog.enabled?"开启":"关闭"} · 为其他程序预留 {dialog.reserve_mib} MiB 显存。</Text>
            <Text>CPU 内存自动回退：{dialog.cpu_fallback?"明确允许":"不允许"}。</Text>
            <Text>Qwen GPU+RAM 权重分层：{dialog.gpu_ram_offload===undefined?(resources.gpu_ram_offload_enabled!==false?"启用":"关闭"):(dialog.gpu_ram_offload?"启用":"关闭")}。仅实验性质，不保证速度。</Text>
            <Alert tone="warning">CPU 内存模式需要关闭显卡上的旧模型进程、重新加载 CPU 权重。
              大模型可能非常慢，并占用数 GB 系统内存。自动降档只使用已下载校验的模型，不自动联网。</Alert>
          </>:null}
          {dialog.kind==="gpu-shortage"?<>
            <Text>当前 GPU 可用 {dialog.free_mib} MiB；所选模型预计需要 {dialog.estimated_mib} MiB，
              还要为游戏及其他应用预留 {reserveMiB} MiB。</Text>
            <Alert tone="warning">继续尝试显存可能出现 CUDA 内存不足、推理失败或影响游戏。
              改用系统内存（CPU）会释放此插件的 GPU 模型，但速度可能明显变慢。</Alert>
            <Text>这只是显存测量的估计；模型加载仍需实际验证。</Text>
          </>:null}
          {dialog.kind==="health-error"?<Alert tone="warning">{short(dialog.error,"自动检测失败，请查看日志",300)}</Alert>:null}
          {dialog.kind==="stem-select"?<Text>先校验所选模型的 SHA-256，通过后停用旧分离进程，并从下一段使用新模型；其他档位文件不会删除。</Text>:null}
          {dialogIsLyrics||dialog.kind==="stem-download"?<>
            <Text>下载大小：约 {mb(dialogPlan.download_bytes_estimate)}</Text>
            <Text>建议预留硬盘：约 {mb(dialogPlan.disk_required_bytes_estimate)}</Text>
            <Text>GPU 模式建议空闲显存（至少）：{dialogPlan.suggested_free_vram_mib||"—"} MiB</Text>
            <Text>使用 CPU 时不需要独立显卡显存；GPU 所需空间会随模型、运行库和并行任务变化。</Text>
            <Text>GPU 预计占用：{short(dialogPlan.estimated_gpu_peak)}</Text>
            <Text>支持 CPU；NVIDIA GPU 需兼容的 CUDA 运行库和实际推理验证。</Text>
            <Text>来源：{source==="mirror"?"国内镜像优先，失败转官方":"官方优先，失败转国内镜像"}</Text>
            <Text>完整 SHA-256 校验通过后才能试运行启用；不自动替换现有歌词模型。</Text>
          </>:null}
          {dialog.kind==="stem-gpu-install"?<>
            <Text>分离计算目前使用 ONNX Runtime CPU，和你已有的 Whisper CUDA 加速不是同一组件。</Text>
            <Text>仅安装在插件数据目录：ONNX Runtime GPU 1.26 + CUDA 12.x + cuDNN 9 + cuBLAS/cuFFT 等依赖。首次总下载约 1.8 GB，建议预留至少 6 GB 磁盘空间。</Text>
            <Text>要求 Windows x64、插件独立 Python 3.11、兼容 NVIDIA 驱动和 VC++ 运行库。CUDA/cuDNN DLL 全部安装在插件环境，不写入系统目录。</Text>
            <Text>来源：{source==="mirror"?"国内镜像优先，失败转官方":"PyPI 官方优先，失败转国内镜像"}；每个 GPU/DLL 安装包使用固定 SHA-256，已下载且完整的文件会直接复用。</Text>
            <Text>下载成功不等于 GPU 已可运行：必须再点「GPU 实际推理检测」。检测失败会保留 CPU 分离，不删除模型。</Text>
          </>:null}
          {dialog.kind==="app-install"?<>
            <Text>将安装 proc-tap 1.1.1（Windows 64 位二进制包，约 126 KB），存放在插件独立环境，不修改系统 Python。</Text>
            <Text>最低：Windows 10 Build 20348 或 Windows 11。CPU 采集，无需 GPU 显存；系统声音仅在再次授权选择应用后才开始采集。</Text>
            <Text>来源：{source==="mirror"?"国内镜像优先，失败尝试 PyPI 官方":"PyPI 官方优先，失败尝试国内镜像"}。</Text>
          </>:null}
          {dialog.kind==="app-select"?<>
            <Text>应用：{short(dialog.name)} · PID {dialog.pid}</Text>
            <Text>模式：{appModeNames[dialog.mode]||"仅监测"}</Text>
            <Text>只采集该进程及其子进程，不录麦克风，也不处理其他应用的声音。该进程内多个声音仍可能混在一起。</Text>
            <Text>切换来源时，旧的歌词与视频时间轴立即失效；不会把旧结果当成当前来源。</Text>
          </>:null}
          {dialog.kind==="pin-host"?<>
            <Text>程序：{short(dialog.name)} · PID {dialog.pid}</Text>
            <Text>保存此程序的完整可执行文件路径，之后即使 PID 变化也自动标成 AI 声音。独立监听其他应用时，不会采集它及识别到的子进程。</Text>
            <Text>此操作不会静音 Windows：你仍能听见 AI。只有 N.E.K.O 宿主内的声音才会被固定排除；系统混音模式不能保证这种隔离。</Text>
          </>:null}
          {dialog.kind==="app-system"?<Text>系统混音会采集所选扬声器上所有应用的声音，可能包含 AI 自身语音。若要确保不听 AI，请选择其他应用的独立声道。</Text>:null}
          {dialog.kind==="channel-share"?<Text>{dialog.enabled===true?
            "授权后，其他获允许的插件可通过 get_selected_app_outputs 读取当前选中应用的文字候选与伴奏/节拍事件；不提供PCM和历史聊天。":
            "关闭后立即拒绝其他插件新的分析摘要读取请求；不会关闭本地识别。"}</Text>:null}
          {dialog.kind==="media-role"?<>
            <Text>来源：{short(dialog.app)}</Text>
            <Text>标记为：{dialog.role==="music"?"音乐":dialog.role==="video"?"视频":dialog.role==="ai"?"AI 语音":"自动"}</Text>
            <Text>此操作只提供二次验证的线索，不能证明该应用正在实际输出声音。</Text>
          </>:null}
          {dialog.kind==="capture-output"?<>
            <Text>选择：{short(dialog.output_name,"系统默认扬声器")}</Text>
            <Text>只监听这个输出设备。需要 N.E.K.O TTS 本身支持独立输出设备才能真正分流；相同程序进程的混合音频不能凭扬声器按钮拆开。</Text>
            <Text>如果 TTS 仍在所选输出，必须接入说话开始/结束通知，否则请手动屏蔽。切换前暂停插件。</Text>
            {dialog.output_id?<div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"8px"}}>
              <Text>我已确认 AI 语音真正路由到其他输出设备</Text>
              <Switch checked={dialog.ai_separate===true}
                onChange={(value:boolean)=>setDialog({...dialog,ai_separate:value})}/>
            </div>:null}
          </>:null}
          {dialog.kind==="source"?<>
            <Text>{dialog.value==="mirror"?"优先尝试国内公开镜像，失败再尝试官方。":"优先尝试 Hugging Face 官方，失败时尝试国内镜像。"}</Text>
            <Text>所有来源都验证官方固定 SHA-256；镜像可访问并不保证文件 CDN 一定可下载。</Text>
          </>:null}
          {dialog.kind==="device"||dialog.kind==="stem-device"?<>
            <Text>{dialog.value==="cpu"?"CPU：无需 NVIDIA 显卡，不占用模型专属显存；速度依设备而异。":
              "NVIDIA GPU：可能加快推理，但需要驱动、CUDA、cuDNN 等兼容；显卡存在不等于可运行。"}</Text>
            <Text>{dialog.kind==="stem-device"?"分离依赖 ONNX Runtime CUDA Provider；不支持时会报告实际设备。":
               "歌词推理仍需在加载/试运行成功后才会真正使用新设备。"}</Text>
          </>:null}
          {dialog.kind==="lyrics-uninstall"||dialog.kind==="stem-uninstall"?<>
            <Alert tone="warning">只删除选中档位。如模型仍被占用，将先自动暂停本插件并关闭分离或歌词子进程；不会暂停系统播放器。卸载后可手动恢复插件。</Alert>
            <Text>Windows 文件锁定时会停止删除并显示错误，不会假报卸载成功；其它模型和环境不受影响。</Text>
          </>:null}
          {dialog.kind==="stem-enable"?<Text>会分别处理人声和伴奏，CPU 开销可能增加；分离失败回退原声音频。</Text>:null}
          {dialogIsLyrics||dialogIsStem?<Text>显存容量为参考估算，最终以真实设备测试为准。</Text>:null}
        </Stack>
        </div>
      </div>
  </Modal>:null}
  </>
}
