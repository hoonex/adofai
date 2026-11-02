/* Browser event inspector: actual chart data is kept local. Supports active BPM,
 * SetSpeed/Twirl/Pause/Hold/ScaleRadius map icons, event lists and flash preview.
 * Other complex Unity effects are surfaced without claiming execution.
 */
(function(){
'use strict';
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const A=()=>window.ADOFAIChartPort,R=()=>window.ADOFAIArchivePlayer;
const H={doc:null,signature:'',current:-1,list:new Map(),last:0};
const safe=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=x=>Number.isInteger(x)?String(x):Number(x).toFixed(2).replace(/0+$/,'').replace(/\.$/,'');
function jump(n,seek=false){
 if(!A()?.state.doc)return;const floor=Math.max(0,Math.min(A().state.angles.length,Number(n)||0));
 A().selectFloor(floor);A().camera?.centerFloor(floor);
 if(seek&&R()?.state.timings?.[floor]!=null){const audio=$('chartAudio');if(audio?.src)audio.currentTime=Math.max(0,R().state.timings[floor]);}
}
function boot(){
 if(!A()||!R()||!document.querySelector('.pc-media')){setTimeout(boot,120);return;}
 const hud=document.createElement('div');hud.id='adoEventHud';
 hud.innerHTML='<div class="ado-live"><strong id="adoActiveBpm">BPM —</strong><b id="adoCurrentMult">×—</b><span id="adoTwirl">↷ 정회전</span><label>연습 재생 <select id="adoRate"><option value=".5">0.5×</option><option value=".75">0.75×</option><option value="1" selected>1.0×</option><option value="1.25">1.25×</option><option value="1.5">1.5×</option><option value="2">2.0×</option></select></label><button id="adoPlaySelected">▶ 선택 타일부터 재생</button><label><input id="adoFlashPreview" type="checkbox" checked>플래시 미리보기</label></div><div id="adoSpeedNav" class="ado-speed-nav">차트를 열면 배속 변경 지점이 표시됩니다.</div><div class="ado-summary"><button id="adoExpandEvents" aria-expanded="false">전체 이벤트 펼치기</button><span id="adoEventStatus">이벤트 파일을 읽고 있습니다.</span></div><div id="adoEventGroups" hidden></div>';
 const stats=document.querySelector('.pc-media .pc-stats');
 if(stats)stats.insertAdjacentElement('afterend',hud);else document.querySelector('.pc-media').append(hud);
 const flash=document.createElement('div');flash.id='adoFlash';flash.setAttribute('aria-hidden','true');document.querySelector('.chart-stage').append(flash);
 const style=document.createElement('style');style.textContent=[
 '#adoEventHud{border-top:1px solid #425665;background:#1b2b38;padding:8px 13px;font-size:12px;color:#d7e4f1}',
 '.ado-live{display:flex;flex-wrap:wrap;align-items:center;gap:8px}',
 '.ado-live>strong{color:#ffdf99;font-size:17px;background:#263b4d;border:1px solid #9b7b45;padding:5px 8px;border-radius:5px}',
 '.ado-live>b,.ado-live>span{padding:5px 8px;border:1px solid #4b687d;background:#22394a;border-radius:4px}',
 '.ado-live label{display:inline-flex;align-items:center;gap:5px}',
 '#adoEventHud button,#adoRate{background:#33495e;color:#fff;border:1px solid #7188a0;border-radius:4px;padding:7px;cursor:pointer;min-height:33px}',
 '.ado-speed-nav{display:flex;overflow:auto;gap:6px;padding:9px 0 3px}',
 '.ado-speed-nav button{white-space:nowrap;flex-shrink:0}',
 '.ado-speed-nav button.current{background:#866a39!important;border-color:#ffcf7a!important}',
 '.ado-summary{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-top:6px;color:#adc4d8}',
 '#adoEventGroups{max-height:330px;overflow:auto;border:1px solid #39556b;background:#111f2c;margin-top:7px;padding:8px;border-radius:4px}',
 '.ado-event-group{margin:5px 0;border:1px solid #36536a;border-radius:4px}',
 '.ado-event-group summary{cursor:pointer;padding:8px}',
 '.ado-event-rows{padding:7px;display:flex;gap:5px;flex-wrap:wrap;max-height:240px;overflow:auto}',
 '.ado-event-rows button{font-size:11px}',
 '#adoFlash{position:absolute;inset:48px 0 55px;pointer-events:none;z-index:6;opacity:0}',
 '.ado-marker{cursor:pointer}.ado-marker text{font:700 11px system-ui,sans-serif;fill:#111820;text-anchor:middle;pointer-events:none}',
 '.ado-marker:focus-visible rect{stroke:#fff;stroke-width:2}',
 '@media(max-width:700px){#adoEventHud{padding:7px}.ado-live{gap:5px}.ado-live>strong{font-size:15px}}'
 ].join('\n');document.head.append(style);
 $('adoRate').onchange=()=>{const a=$('chartAudio');if(a)a.playbackRate=Number($('adoRate').value)||1;};
 $('adoPlaySelected').onclick=()=>R().startPlay();
 $('adoExpandEvents').onclick=()=>{
  const n=$('adoEventGroups');n.hidden=!n.hidden;
  $('adoExpandEvents').textContent=n.hidden?'전체 이벤트 펼치기':'전체 이벤트 접기';
  $('adoExpandEvents').setAttribute('aria-expanded',String(!n.hidden));
 };
 window.addEventListener('adofai:timing-updated',()=>{renderSpeed();renderMarkers(true);});
 new MutationObserver(()=>renderMarkers()).observe($('chartSvg'),{childList:true});
 requestAnimationFrame(tick);
}
function renderSpeed(){
 const p=R()?.state,doc=A()?.state.doc;if(!doc||!p)return;
 const base=Math.max(1,Number(doc.settings?.bpm)||120);
 const rows=[{floor:0,bpm:base,multiplier:1}].concat(p.speedChanges||[]);
 $('adoSpeedNav').innerHTML=rows.map((x,i)=>'<button data-floor="'+x.floor+'" data-seq="'+i+'" title="타일 '+x.floor+'번">'+(i?'타일 '+x.floor:'시작')+' · '+fmt(x.bpm)+' BPM (×'+fmt(x.multiplier)+')</button>').join('');
 for(const b of $('adoSpeedNav').querySelectorAll('button'))b.onclick=()=>jump(b.dataset.floor,true);
 const counts=new Map();H.list.clear();
 for(const e of doc.actions||[]){
  counts.set(e.eventType,(counts.get(e.eventType)||0)+1);
  if(!Number.isInteger(e.floor))continue;
  const group=H.list.get(e.floor)||[];group.push(e);H.list.set(e.floor,group);
 }
 const entries=[...counts.entries()].sort((a,b)=>b[1]-a[1]);
 $('adoEventGroups').innerHTML=entries.map(([name,count])=>{
  const items=(doc.actions||[]).filter(e=>e.eventType===name);
  const level=['SetSpeed','Twirl','Pause'].includes(name)?'타이밍 연결':name==='Hold'?'부분 지원':name==='Flash'?'간단 미리보기':'표시·편집 가능 / 원본 효과 미구현';
  return '<details class="ado-event-group"><summary>'+safe(name)+' · '+count+'개 · '+level+'</summary><div class="ado-event-rows">'+
    items.map(e=>'<button data-floor="'+e.floor+'">'+e.floor+'번</button>').join('')+'</div></details>';
 }).join('');
 for(const b of $('adoEventGroups').querySelectorAll('button[data-floor]'))b.onclick=()=>jump(b.dataset.floor,true);
 $('adoEventStatus').textContent='이벤트 '+(doc.actions||[]).length+'개 · '+entries.length+'종류 · 현재 BPM / 배속 / 회전 / 정지 표시';
}
function renderMarkers(force=false){
 const svg=$('chartSvg'),doc=A()?.state.doc;if(!svg||!doc)return;
 if(svg.querySelector('#adoMarkers')&&!force)return;
 svg.querySelector('#adoMarkers')?.remove();
 const group=document.createElementNS(NS,'g');group.id='adoMarkers';group.style.pointerEvents='all';
 const grouped=new Map();
 for(const e of doc.actions||[]){
  if(!['SetSpeed','Twirl','Pause','Hold','ScaleRadius','MultiPlanet'].includes(e.eventType))continue;
  if(!Number.isInteger(e.floor)||!A().state.positions[e.floor])continue;
  const a=grouped.get(e.floor)||[];if(!a.some(x=>x.eventType===e.eventType))a.push(e);
  grouped.set(e.floor,a);
 }
 const map=new Map((R()?.state.speedChanges||[]).map(e=>[e.floor,e]));
 for(const [floor,items] of grouped){
  const pos=A().state.positions[floor];let idx=0;
  for(const e of items){
   const sp=map.get(floor);
   const label=e.eventType==='SetSpeed'?(sp?'×'+fmt(sp.multiplier):'BPM'):({Twirl:'↶',Pause:'Ⅱ',Hold:'H',ScaleRadius:'R',MultiPlanet:'3P'})[e.eventType];
   const color=({SetSpeed:'#ffd179',Twirl:'#d4bbf8',Pause:'#ffaab1',Hold:'#a1ecc1',ScaleRadius:'#afccfb',MultiPlanet:'#9ce9ef'})[e.eventType];
   const w=Math.max(27,label.length*8+13);
   const x=pos.x*54+idx*22-13,y=pos.y*54-44-Math.floor(idx/2)*24;idx++;
   const b=document.createElementNS(NS,'g');b.setAttribute('class','ado-marker');b.setAttribute('role','button');b.setAttribute('tabindex','0');b.setAttribute('data-floor',String(floor));b.setAttribute('aria-label',e.eventType+' '+floor+'번 타일');
   const rect=document.createElementNS(NS,'rect');for(const [key,v] of Object.entries({x:x-w/2,y:y-12,width:w,height:23,rx:5,fill:color,stroke:'#ffffff88'}))rect.setAttribute(key,String(v));
   const t=document.createElementNS(NS,'text');t.setAttribute('x',String(x));t.setAttribute('y',String(y+4));t.textContent=label;
   b.append(rect,t);b.onclick=()=>jump(floor);b.onkeydown=ev=>{if(ev.key==='Enter'){jump(floor);ev.preventDefault();}};
   group.append(b);
  }
 }
 const planets=svg.querySelector('#adoPlanets');if(planets)planets.before(group);else svg.append(group);
}
function updateFlash(floor,time){
 const box=$('adoFlash');if(!box)return;
 if(!$('adoFlashPreview').checked){box.style.opacity=0;return;}
 const doc=A()?.state.doc,p=R()?.state;if(!doc||!p)return;
 let match=null;
 for(const e of doc.actions||[]){
  if(e.eventType!=='Flash'||!Number.isInteger(e.floor)||e.floor>floor)continue;
  const start=p.timings?.[e.floor];if(start==null)continue;
  const dur=Math.max(.03,Number(e.duration||0)*60/(p.bpmAt?.[e.floor]||120));
  const ratio=(time-start)/dur;if(ratio>=0&&ratio<=1)match={e,ratio};
 }
 if(!match){box.style.opacity=0;return;}
 const e=match.e,col=String(e.startColor||'ffffff').replace('#','');
 box.style.background=/^[a-f\d]{6,8}$/i.test(col)?'#'+col.slice(0,6):'#ffffff';
 const start=Number(e.startOpacity||0),end=Number(e.endOpacity||0);
 box.style.opacity=String(Math.max(0,Math.min(.55,(start+(end-start)*match.ratio)*.006)));
}
function tick(ts){
 if(ts-H.last>140){H.last=ts;
  const chart=A()?.state.doc,p=R()?.state;
  if(chart&&p){
   const sig=[chart.actions?.length||0,chart.settings?.bpm,A().state.angles?.length].join('|');
   if(chart!==H.doc||sig!==H.signature){
    H.doc=chart;H.signature=sig;H.current=-1;R().updateTiming();renderSpeed();renderMarkers(true);
   }
   renderMarkers();
   const f=p.playing?p.playerIndex:A().state.selected;
   const bpm=p.bpmAt?.[f]||Math.max(1,Number(chart.settings?.bpm)||120);
   const base=p.baseBpm||Math.max(1,Number(chart.settings?.bpm)||120);
   $('adoActiveBpm').textContent='BPM '+fmt(bpm);
   $('adoCurrentMult').textContent='맵 배속 ×'+fmt(bpm/base);
   $('adoTwirl').textContent=p.twirlAt?.[f]?'↶ 역회전':'↷ 정회전';
   const speeds=(p.speedChanges||[]),rows=$('adoSpeedNav').querySelectorAll('button');
   for(let i=0;i<rows.length;i++)rows[i].classList.toggle('current',f>=Number(rows[i].dataset.floor)&&f<(speeds[i]?.floor??Infinity));
   if(f!==H.current){
    H.current=f;const events=H.list.get(f)||[];
    $('adoEventStatus').textContent='현재 '+f+'번 타일 · '+events.length+'개 이벤트'+(events.length?' · '+[...new Set(events.map(e=>e.eventType))].slice(0,7).join(', '):'');
   }
   const clock=p.playing?$('chartAudio')?.currentTime:p.timings?.[f];if(typeof clock==='number')updateFlash(f,clock);
  }
 }
 requestAnimationFrame(tick);
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();