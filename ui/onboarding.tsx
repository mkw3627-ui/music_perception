import {
  Card, Stack, Text, Button, Progress, List, Alert, Field, Input,
  RadioGroup, SegmentedControl, StatusBadge, Divider,
  useState, useEffect,
} from "@neko/plugin-ui"
import type { PluginSurfaceProps } from "@neko/plugin-ui"

// Dashboard 里的 onboarding 上下文（契约见方案 3.1）。
export type OnboardingState = any

function unpack(response:any):any {
  let data = response
  for (let i=0; i<6; i++) {
    if (!data || typeof data !== "object") break
    if (data.success === false) throw Error(String(data.error || "操作失败"))
    if (data.value !== undefined) {data=data.value;continue}
    if (data.data !== undefined && data.ok === undefined) {data=data.data;continue}
    if (data.result !== undefined && data.ok === undefined) {data=data.result;continue}
    break
  }
  if (data && data.ok === false) throw Error(String(data.error || data.message || "操作失败"))
  return data || {}
}
function short(value:any, fallback="—", maxChars=90):string {
  if (value===null || value===undefined || value==="") return fallback
  const s=String(value)
  return s.length>maxChars ? s.slice(0,maxChars)+"…" : s
}
function fmtBytes(n:any):string {
  const v=Number(n||0)
  if (!v) return "—"
  if (v>=1024*1024*1024) return (v/1024/1024/1024).toFixed(1)+" GB"
  if (v>=1024*1024) return Math.round(v/1024/1024)+" MB"
  return Math.round(v/1024)+" KB"
}
function failureReason(job:any):string {
  const list = Array.isArray(job?.attempts) ? job.attempts : []
  for (let i=list.length-1; i>=0; i--) {
    const a = list[i]
    if (a && a.ok === false && a.error) return String(a.error)
  }
  return job?.error ? String(job.error) : "下载失败"
}

// 档位定义只保留一份：panel.tsx 从这里导入。
export const singingOptions = [
  {value:"low",label:"低",name:"Whisper Small",description:"最省资源，只适合安静独唱"},
  {value:"medium",label:"中",name:"Qwen3-ASR 0.6B",description:"中英文都行"},
  {value:"high",label:"高",name:"Whisper Large-v3",description:"是细致识别首选"},
  {value:"ultra",label:"极高",name:"Qwen3-ASR 1.7B",description:"最准，但显存吃掉约 5%"},
]
export const qualities = [
  {value:"low", label:"低", model:"tiny", name:"Tiny"},
  {value:"medium", label:"中", model:"base", name:"Base"},
  {value:"high", label:"高", model:"small", name:"Small"},
  {value:"ultra", label:"极高", model:"large-v3-turbo", name:"Turbo"},
]
export const stemChoices = [
  {value:"low",label:"低",name:"Karaoke 2",hint:"约 53MB · 实际效果按歌曲试听"},
  {value:"medium",label:"中",name:"Inst HQ2",hint:"另一个伴奏分离权重"},
  {value:"high",label:"高",name:"Vocal FT",hint:"以人声为目标的分离"},
  {value:"ultra",label:"极高",name:"Inst HQ3",hint:"兼容旧版模型，不保证每首都最佳"},
]

const EARS = [
  {icon:"♪", name:"听歌", desc:"认出正在放的歌和歌词"},
  {icon:"▶", name:"看视频", desc:"听懂视频里在说什么"},
  {icon:"◎", name:"应用声道", desc:"只听你指定的那个软件"},
  {icon:"⇄", name:"人声分离", desc:"把人声和伴奏拆开"},
  {icon:"◍", name:"唱歌识别", desc:"听清歌里唱的词"},
  {icon:"≈", name:"律动", desc:"角色跟着音乐摆"},
]
const STEPS = ["认识", "环境", "位置", "配置", "下载"]

export function Onboarding(props:{ob:OnboardingState, api:any, onClose:()=>void,
                                   startStep?:number}) {
  const ob = props.ob || {}
  const [step, setStep] = useState<number>(props.startStep || Number(ob.step||1) || 1)
  const [busy, setBusy] = useState<string>("")
  const [error, setError] = useState<string>("")
  const [manualPath, setManualPath] = useState<string>("")
  const [mode, setMode] = useState<string>("recommend")
  const [videoQ, setVideoQ] = useState<string>("")
  const [singQ, setSingQ] = useState<string>("")
  const [stemQ, setStemQ] = useState<string>("")
  const [device, setDevice] = useState<string>("")
  const [source, setSource] = useState<string>(ob.source_pref || "mirror")
  const [browsing, setBrowsing] = useState<boolean>(false)

  const call = (id:string, args:any, okNotice?:string) => {
    setBusy(id); setError("")
    return props.api.call(id, args||{}, {userInitiated:true})
      .then(unpack)
      .then((out:any)=>{ if (okNotice) setNotice(okNotice); return out })
      .catch((e:any)=>{ setError(short(e?.message || e, "操作失败", 160)); return {} })
      .finally(()=>{ setBusy(""); void props.api.refresh() })
  }
  const [notice, setNotice] = useState<string>("")

  const scan = ob.scan || {}
  const items:any[] = Array.isArray(scan.items) ? scan.items : []
  const check = ob.storage_check || {}
  const hw = ob.hardware || {}
  const rec = ob.recommend || {}
  const picker = ob.picker || {}
  const jobs:any[] = Array.isArray(ob.jobs) ? ob.jobs : []

  // 有任务在跑时轮询刷新（沿用 panel.tsx 的写法）。
  useEffect(()=>{
    const running = jobs.some((j:any)=>j && (j.state==="running"||j.state==="queued"))
    if (!running) return
    const t = setInterval(()=>{ void props.api.refresh() }, 1500)
    return ()=>clearInterval(t)
  }, [jobs.length, jobs.map((j:any)=>j&&j.state).join(",")])

  const goScan = ()=>call("onboarding_scan", {}, "").then(()=>setStep(2))
  const browse = ()=>{
    setBrowsing(true)
    call("onboarding_browse_start", {start_path: ob.storage_root || ob.default_root || ""}, "")
    // 非阻塞对话框：前端等待，直到状态不再是 open。
    const poll = ()=>{
      props.api.call("onboarding_browse_wait", {seconds:15}, {userInitiated:true})
        .then(unpack).then((out:any)=>{
          const p = out.picker || out
          if (p && p.state === "open") { window.setTimeout(poll, 800); return }
          setBrowsing(false)
          if (p && p.state === "done" && p.path) {
            setManualPath(String(p.path))
            void call("onboarding_set_storage_root", {path:String(p.path), confirmed:true}, "")
          }
        }).catch(()=>{ setBrowsing(false) })
    }
    window.setTimeout(poll, 800)
  }

  const plannedIds = ():string[] => {
    const need = items.filter((it:any)=>it.required && it.state!=="ok").map((it:any)=>it.id)
    const chosen:string[] = []
    if (mode==="recommend") {
      if (rec.video_quality) chosen.push("lyrics_model")
      if (rec.singing_quality) chosen.push("singing")
      if (rec.stem_quality) chosen.push("stem_model")
    } else {
      if (videoQ && videoQ!=="off") chosen.push("lyrics_model")
      if (singQ && singQ!=="off") chosen.push("singing")
      if (stemQ && stemQ!=="off") chosen.push("stem_model")
    }
    return Array.from(new Set([...need, ...chosen]))
  }

  const startInstall = ()=>{
    const picked = plannedIds()
    call("onboarding_start_install", {
      items: picked,
      choices: mode==="recommend"
        ? {video_quality:rec.video_quality||"high", singing_quality:rec.singing_quality||"medium",
           stem_quality:rec.stem_quality||"low", device:rec.device||"cpu"}
        : {video_quality:videoQ||"high", singing_quality:singQ||"medium",
           stem_quality:stemQ||"low", device:device||"cpu"},
      confirmed:true,
    }, "开始下载").then(()=>setStep(5))
  }

  const planned = plannedIds()
  const totalBytes = items.reduce((a:number,it:any)=>
    a+(planned.indexOf(String(it.id))>=0?Number(it.size_bytes||0):0),0)
  const freeOk = !check.free_bytes || Number(check.free_bytes) > totalBytes
  const vramNeed = device==="cuda"||rec.device==="cuda" ? Number(hw.reserve_mib||0) : 0
  const vramOk = !hw.gpu_free_mib || Number(hw.gpu_free_mib) > vramNeed
  const canNext = step!==3 || check.ok===true

  return (
    <Stack>
      {/* 步骤指示条 */}
      <div style={{display:"flex",gap:"6px"}}>
        {STEPS.map((name:string,i:number)=>{
          const n=i+1
          const active=n===step, done=n<step
          return <div key={name} style={{flex:1,minWidth:0}}>
            <div style={{height:"4px",borderRadius:"4px",
              background:active?"#2db6a4":done?"#2db6a455":"#ffffff17"}}/>
            <div style={{fontSize:"10px",marginTop:"5px",
              color:active?"#2db6a4":"#adcee1"}}>{name}</div>
          </div>
        })}
      </div>

      {error?<Alert tone="warning">{error}</Alert>:null}
      {notice?<Alert tone="info">{notice}</Alert>:null}

      {step===1?<Card title="认识她的耳朵"><Stack>
        <Text>这一步先看看她能听见什么。</Text>
        <div style={{display:"grid",gridTemplateColumns:"repeat(3,minmax(0,1fr))",gap:"9px"}}>
          {EARS.map((e:any)=>
            <div key={e.name} style={{borderRadius:"11px",background:"#ffffff0b",
              border:"1px solid #ffffff17",padding:"10px",minWidth:0}}>
              <div style={{fontSize:"18px",color:"#2db6a4"}}>{e.icon}</div>
              <div style={{fontSize:"13px",marginTop:"4px",fontWeight:750}}>{e.name}</div>
              <div style={{fontSize:"11px",color:"#adcee1",marginTop:"3px"}}>{e.desc}</div>
            </div>)}
        </div>
        <Button disabled={Boolean(busy)} onClick={goScan}>
          {busy==="onboarding_scan"?"检测中…":"开始检测"}</Button>
      </Stack></Card>:null}

      {step===2?<Card title="检查环境"><Stack>
        {scan.state==="running"?<Progress label="正在检查…"/>:null}
        <List items={items.map((it:any)=>({
          title:`${it.state==="ok"?"✓":it.state==="optional"?"○":"✕"} ${short(it.name,"项目",30)}`,
          description:`${short(it.purpose,"",40)} · ${fmtBytes(it.size_bytes)}`,
          tone: it.state==="ok"?"success":it.state==="optional"?"neutral":"warning",
        }))}/>
        {items.length===0?<Text>还没有检查结果。</Text>:null}
        <div style={{display:"flex",gap:"8px"}}>
          <Button disabled={Boolean(busy)} onClick={goScan}>重新检测</Button>
          <Button disabled={Boolean(busy)} tone="primary"
            onClick={()=>setStep(items.every((it:any)=>!it.required||it.state==="ok")?4:3)}>
            下一步</Button>
        </div>
      </Stack></Card>:null}

      {step===3?<Card title="选位置"><Stack>
        <Text>模型和依赖要放在这里。</Text>
        <Field label="位置">
          <Input value={manualPath||ob.storage_root||ob.default_root||""}
            onChange={(v:string)=>setManualPath(v)} placeholder="D:\…\data"/>
        </Field>
        <Text>可用空间：{fmtBytes(check.free_bytes)}</Text>
        {check.ok?<StatusBadge tone="success">可以使用</StatusBadge>:
          check.error?<Alert tone="warning">{short(check.error,"",120)}</Alert>:
          <StatusBadge tone="neutral">还没确认</StatusBadge>}
        {Array.isArray(check.warnings)&&check.warnings.length?
          <Alert tone="warning">{short(check.warnings.join("；"),"",160)}</Alert>:null}
        <div style={{display:"flex",gap:"8px"}}>
          <Button disabled={Boolean(busy)} onClick={browse}>
            {browsing?"等待选择…":"选择文件夹…"}</Button>
          {(browsing||picker.state==="open")?<Button onClick={()=>{
            setBrowsing(false); void call("onboarding_browse_cancel",{},"")}}>取消</Button>:null}
          <Button disabled={Boolean(busy)||!String(manualPath||ob.storage_root||"").trim()}
            onClick={()=>call("onboarding_set_storage_root",
              {path:String(manualPath||ob.storage_root||""), confirmed:true}, "位置已确认")}>
            使用这个位置</Button>
        </div>
        <Button disabled={Boolean(busy)||!canNext} tone="primary" onClick={()=>setStep(4)}>
          下一步</Button>
      </Stack></Card>:null}

      {step===4?<Card title="选配置"><Stack>
        <RadioGroup value={mode} options={[{value:"recommend",label:"推荐"},{value:"custom",label:"自定义"}]}
          onChange={(v:string)=>setMode(v)}/>
        {mode==="recommend"?<Stack>
          <Text>{short(rec.reason,"按你的电脑自动选",120)}</Text>
          <Text>视频语音 {rec.video_quality||"—"} · 唱歌识别 {rec.singing_quality||"—"} · 人声分离 {rec.stem_quality||"—"}</Text>
          <Text>设备 {rec.device==="cuda"?"NVIDIA GPU":"CPU"}</Text>
        </Stack>:<Stack>
          <Text>视频语音</Text>
          <SegmentedControl value={videoQ||rec.video_quality||"high"}
            options={[...qualities.map((q:any)=>({value:q.value,label:q.label})),
                      {value:"off",label:"不装"}]}
            onChange={(v:string)=>setVideoQ(v)}/>
          <Text>唱歌识别</Text>
          <SegmentedControl value={singQ||rec.singing_quality||"medium"}
            options={[...singingOptions.map((q:any)=>({value:q.value,label:q.label})),
                      {value:"off",label:"不装"}]}
            onChange={(v:string)=>setSingQ(v)}/>
          <Text>人声分离</Text>
          <SegmentedControl value={stemQ||rec.stem_quality||"low"}
            options={[...stemChoices.map((q:any)=>({value:q.value,label:q.label})),
                      {value:"off",label:"不装"}]}
            onChange={(v:string)=>setStemQ(v)}/>
          <Text>设备</Text>
          <SegmentedControl value={device||rec.device||"cpu"}
            options={[{value:"cpu",label:"CPU"},{value:"cuda",label:"GPU"}]}
            onChange={(v:string)=>setDevice(v)}/>
        </Stack>}
        <Divider/>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"8px"}}>
          <Text>位置 {short(ob.storage_root||ob.default_root,"未设置",60)}</Text>
          <Button onClick={()=>setStep(3)}>修改位置</Button>
        </div>
        <Divider/>
        <Text>下载源</Text>
        <SegmentedControl value={source}
          options={[{value:"mirror",label:"自动 · 国内优先"},{value:"official",label:"国外优先"}]}
          onChange={(v:string)=>{setSource(v);
            call("set_download_source",{source:v,confirmed:true},"下载源已切换")}}/>
        <Text>合计 {fmtBytes(totalBytes)} · 磁盘{freeOk?"够":"可能不够"} · 显存{vramOk?"够":"可能不够"}</Text>
        {(!freeOk||!vramOk)?<Alert tone="warning">空间可能不够，仍可继续。</Alert>:null}
        <Button disabled={Boolean(busy)} tone="primary" onClick={startInstall}>开始下载</Button>
      </Stack></Card>:null}

      {step===5?<Card title="下载"><Stack>
        {jobs.length===0?<Text>还没有任务。</Text>:null}
        {jobs.map((job:any,i:number)=>
          <Card key={job?.id||i} title={short(job?.label,"任务",24)}>
            <Stack>
              <Text>{short(job?.state==="done"?"完成":job?.state==="failed"?"失败":
                job?.state==="cancelled"?"已取消":"进行中","",20)}</Text>
              <Progress value={job?.indeterminate?undefined:Number(job?.progress_pct||0)}
                label={short(job?.stage,"",40)}/>
              <Text>{fmtBytes(job?.bytes)} / {fmtBytes(job?.total_bytes)}
                {job?.speed_bytes_s?` · ${fmtBytes(job.speed_bytes_s)}/s`:""}
                {job?.eta_seconds?` · 剩 ${Math.round(Number(job.eta_seconds))} 秒`:""}
                {job?.source?` · ${short(job.source,"",20)}`:""}</Text>
              {Array.isArray(job?.attempts)&&job.attempts.some((a:any)=>a&&a.ok===false)?
                <Text>{job.attempts.filter((a:any)=>a&&a.ok===false)
                  .map((a:any)=>`${short(a.source,"源",12)} 失败`).join("，")}
                  {" → 已切到 "}{short(job.source,"当前源",20)}</Text>:null}
              {job?.state==="failed"?<Stack>
                <Alert tone="warning">{short(failureReason(job),"下载失败",100)}</Alert>
                <Button onClick={()=>call("onboarding_start_install",
                  {items:[job.id],choices:mode==="recommend"
                    ?{video_quality:rec.video_quality||"high",singing_quality:rec.singing_quality||"medium",
                      stem_quality:rec.stem_quality||"low",device:rec.device||"cpu"}
                    :{video_quality:videoQ||"high",singing_quality:singQ||"medium",
                      stem_quality:stemQ||"low",device:device||"cpu"},confirmed:true},"重试中")}>
                  重试</Button>
              </Stack>:null}
            </Stack>
          </Card>)}
        {jobs.some((j:any)=>j&&(j.state==="running"||j.state==="queued"))?
          <Button onClick={()=>call("onboarding_cancel_install",{},"已取消")}>取消</Button>:null}
        {jobs.length>0&&jobs.every((j:any)=>!j||j.state==="done"||j.state==="cancelled")?
          <Button tone="primary"
            onClick={()=>call("onboarding_finish",{},"设置完成").then(()=>props.onClose())}>
            开始使用</Button>:null}
      </Stack></Card>:null}
    </Stack>
  )
}
