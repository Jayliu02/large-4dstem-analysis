"""Offline canvas review UI; no remote assets or automatic expert decisions."""

HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIB 实测峰复核</title>
<style>
body{font:15px system-ui,sans-serif;margin:24px;background:#f4f6f8;color:#17252d}h1{font-size:24px}
main{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}.viewer{max-width:768px;flex:1 1 500px}
canvas{width:100%;height:auto;background:#111;cursor:crosshair}aside{flex:1 1 300px;max-width:480px}
button,input,select{font:inherit;margin:4px;padding:8px}button{cursor:pointer}button:disabled{cursor:default}
.panel{background:white;padding:14px;border-radius:8px;margin:10px 0}.muted{color:#556570;font-size:13px}
#status{white-space:pre-wrap;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}td,th{padding:5px;border-bottom:1px solid #ddd}
#peaks{max-height:420px;overflow:auto}.warn{color:#963900}label{display:inline-block;margin:5px}
</style>
<h1>MIB 实测峰复核 · 第四轮</h1>
<p>绿色＝有效；红色＝假峰；黄色＝不确定；白色＝未复核。青色十字＝人工补记的主要漏峰。粗圆圈表示两份半计数数据均检出，<b>不代表该峰已被确认</b>。</p>
<main><section class="viewer">
<div class="panel"><button id="prev">上一张</button><select id="pattern"></select><button id="next">下一张</button>
<label><input type="checkbox" id="log" checked>log 显示</label></div>
<canvas id="canvas" width="768" height="768"></canvas>
<p class="muted">点击圈内峰循环标记；Shift+点击图像添加主要漏峰。漏峰坐标为探测器像素。显示截断于每张图 99.7 分位，标记使用原始数值，不能只凭显示亮度判断。</p>
</section><aside>
<div class="panel"><label>复核人 <input id="reviewer" placeholder="填写真实姓名或可追溯标识"></label>
<p id="progress"></p><p id="sample"></p>
<button id="allvalid">当前图全部标为有效</button><button id="clear">重置当前图标注</button>
<label><input type="checkbox" id="complete">当前 DP 已完成复核</label>
<p class="warn">完成前请检查全部峰和主要漏峰；“全部有效”会覆盖当前图已有的逐峰标记。</p>
</div>
<div class="panel"><div id="peaks"></div><h3>主要漏峰</h3><div id="missed"></div></div>
<div class="panel"><button id="save">保存本地草稿</button><button id="export">导出复核 JSON</button>
<label>加载 JSON <input type="file" id="import" accept="application/json,.json"></label>
<p class="muted">草稿只在本浏览器中；换电脑或清理浏览器前请导出 JSON。导出不会自动修改科学验收状态。</p>
<p id="status"></p><p class="muted">导入验收命令：<code>python scripts/06_mib_round4.py --import-review 你的文件.json</code></p>
</div></aside></main>
<script id="bundle" type="application/json">__BUNDLE__</script>
<script>
'use strict';
const bundle=JSON.parse(document.getElementById('bundle').textContent), patterns=bundle.patterns;
const labels=['unreviewed','valid','false_peak','uncertain'], names={unreviewed:'未复核',valid:'有效',false_peak:'假峰',uncertain:'不确定'};
const colors={unreviewed:'#ffffff',valid:'#40ed80',false_peak:'#ff4848',uncertain:'#ffd84c'};
const el=id=>document.getElementById(id), key='mib-review-'+bundle.bundle_id;
let index=0, image=new Image(), token=0;
const fresh=p=>({id:p.id,completed:false,peaks:p.peaks.map(k=>({id:k.id,label:'unreviewed'})),missed_major_peaks:[]});
let state={schema_version:1,bundle_id:bundle.bundle_id,reviewer:'',reviewed_at_utc:null,patterns:patterns.map(fresh)};
function message(t){el('status').textContent=t;}
function validateDraft(s){
 if(s.schema_version!==1||s.bundle_id!==bundle.bundle_id||!Array.isArray(s.patterns)||s.patterns.length!==patterns.length)throw Error('复核来源或图像数量不匹配');
 if(typeof s.reviewer!=='string')throw Error('复核人格式错误');
 const ids=new Set();
 for(const r of s.patterns){const p=patterns.find(p=>p.id===r.id);if(!p||ids.has(r.id))throw Error('未知或重复图像');ids.add(r.id);
  if(typeof r.completed!=='boolean'||!Array.isArray(r.peaks)||r.peaks.length!==p.peaks.length)throw Error('图像或峰记录不完整');
  const seen=new Set();for(const k of r.peaks){if(!Number.isInteger(k.id)||!p.peaks.some(q=>q.id===k.id)||seen.has(k.id)||!labels.includes(k.label))throw Error('峰标识或标签错误');seen.add(k.id);}
  if(r.completed&&r.peaks.some(k=>k.label==='unreviewed'))throw Error('已完成图像仍有未复核峰');
  if(!Array.isArray(r.missed_major_peaks)||r.missed_major_peaks.some(k=>!Number.isFinite(k.x)||!Number.isFinite(k.y)||k.x<0||k.y<0||k.x>=p.width||k.y>=p.height))throw Error('漏峰坐标越界');
 }
 return s;
}
function current(){return state.patterns.find(r=>r.id===patterns[index].id);}
function save(silent=false){state.reviewer=el('reviewer').value;try{localStorage.setItem(key,JSON.stringify(state));if(!silent)message('本地草稿已保存。');}catch(e){message('本地存储不可用，请导出 JSON 保存。');}}
function refresh(){
 const p=patterns[index],r=current();el('pattern').value=String(index);el('prev').disabled=index===0;el('next').disabled=index===patterns.length-1;
 el('complete').checked=r.completed;el('sample').textContent=p.id+' · '+p.peaks.length+' 个检出峰';
 el('progress').textContent='完整复核 '+state.patterns.filter(r=>r.completed).length+' / '+patterns.length+' 张；自动诊断与人工结论分别记录。';
 el('peaks').replaceChildren();const table=document.createElement('table');
 const head=document.createElement('tr');for(const t of ['峰 ID','局部 SNR','标记']){const th=document.createElement('th');th.textContent=t;head.append(th);}table.append(head);
 for(const k of p.peaks){const row=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td'),c=document.createElement('td'),s=document.createElement('select');
  a.textContent=k.id;b.textContent=k.snr.toFixed(1);for(const v of labels){const o=document.createElement('option');o.value=v;o.textContent=names[v];s.append(o);}s.value=r.peaks.find(q=>q.id===k.id).label;
  s.onchange=()=>{r.peaks.find(q=>q.id===k.id).label=s.value;r.completed=false;save(true);refresh();};c.append(s);row.append(a,b,c);table.append(row);
 }el('peaks').append(table);
 el('missed').replaceChildren();r.missed_major_peaks.forEach((m,j)=>{const div=document.createElement('div'),button=document.createElement('button');div.textContent='('+m.x.toFixed(1)+', '+m.y.toFixed(1)+') ';button.textContent='删除';button.onclick=()=>{r.missed_major_peaks.splice(j,1);r.completed=false;save(true);refresh();};div.append(button);el('missed').append(div);});
 const own=++token;image=new Image();image.onload=()=>{if(own===token)draw();};image.src=p[el('log').checked?'image_log':'image_linear'];
}
function draw(){const p=patterns[index],r=current(),c=el('canvas'),ctx=c.getContext('2d'),sx=c.width/p.width,sy=c.height/p.height;ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(image,0,0,c.width,c.height);
 for(const k of p.peaks){ctx.beginPath();ctx.strokeStyle=colors[r.peaks.find(q=>q.id===k.id).label];ctx.lineWidth=k.both_halves?2.5:1;ctx.arc((k.x+.5)*sx,(k.y+.5)*sy,7,0,2*Math.PI);ctx.stroke();ctx.font='10px sans-serif';ctx.fillStyle=ctx.strokeStyle;ctx.fillText(String(k.id),(k.x+.5)*sx+8,(k.y+.5)*sy);}
 ctx.strokeStyle='#00eaff';ctx.lineWidth=2;for(const k of r.missed_major_peaks){const x=(k.x+.5)*sx,y=(k.y+.5)*sy;ctx.beginPath();ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7);ctx.stroke();}
}
patterns.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+p.id;el('pattern').append(o);});
el('pattern').onchange=()=>{index=Number(el('pattern').value);refresh();};el('prev').onclick=()=>{if(index>0){index--;refresh();}};el('next').onclick=()=>{if(index+1<patterns.length){index++;refresh();}};el('log').onchange=refresh;
el('canvas').onclick=e=>{const p=patterns[index],r=current(),rect=el('canvas').getBoundingClientRect(),x=(e.clientX-rect.left)*p.width/rect.width-.5,y=(e.clientY-rect.top)*p.height/rect.height-.5;
 if(e.shiftKey){if(x>=0&&y>=0&&x<p.width&&y<p.height)r.missed_major_peaks.push({x,y});}
 else{let nearest=null,d=4;for(const k of p.peaks){const dist=Math.hypot(k.x-x,k.y-y);if(dist<d){nearest=k;d=dist;}}if(nearest){const item=r.peaks.find(q=>q.id===nearest.id);item.label=labels[(labels.indexOf(item.label)+1)%labels.length];}}
 r.completed=false;save(true);refresh();};
el('allvalid').onclick=()=>{current().peaks.forEach(k=>k.label='valid');current().completed=false;save(true);refresh();};
el('clear').onclick=()=>{const id=patterns[index].id;state.patterns[state.patterns.findIndex(r=>r.id===id)]=fresh(patterns[index]);save(true);refresh();};
el('complete').onchange=()=>{const r=current();if(el('complete').checked&&r.peaks.some(k=>k.label==='unreviewed')){message('请先处理当前图所有峰的标记。');el('complete').checked=false;return;}r.completed=el('complete').checked;save(true);refresh();};
el('reviewer').onchange=()=>save(true);el('save').onclick=()=>save();
el('export').onclick=()=>{state.reviewer=el('reviewer').value.trim();if(!state.reviewer){message('请填写可追溯的复核人标识后导出。');return;}state.reviewed_at_utc=new Date().toISOString();try{validateDraft(state);}catch(e){message(e.message);return;}
 const url=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='mib-review-'+bundle.bundle_id.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);save(true);message('已导出。请用命令行导入；完整复核不一定通过预设阈值。');};
el('import').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;state=validateDraft(JSON.parse(await file.text()));el('reviewer').value=state.reviewer;save(true);refresh();message('标注已加载；尚未执行命令行验收。');}catch(e){message('加载失败：'+e.message);}};
try{const saved=localStorage.getItem(key);if(saved)state=validateDraft(JSON.parse(saved));}catch(e){message('本地草稿不可用：'+e.message);}
el('reviewer').value=state.reviewer;refresh();
</script></html>'''
