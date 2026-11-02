/* Low-latency browser hitsound playback for the PC-scene-linked editor.
 * Uses chart settings.hitsound / hitsoundVolume and SetHitsound events.
 * Procedurally synthesized approximations; original copyrighted Unity
 * audio clips are NOT distributed in this public web bundle.
 */
(function(){
'use strict';
const $=id=>document.getElementById(id);
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const ENGINE={
 context:null,buffers:new Map(),eventFloor:[],eventType:[],eventVolume:[],
 preset:'Kick',volume:100,enabled:true,ready:false,lastPlayed:null,
};
const normalizedType=t=>String(t||'Kick').trim()||'Kick';
function sampleCategory(raw){
 const v=normalizedType(raw).toLowerCase().replace(/[^a-z0-9]/g,'');
 if(v==='none'||v==='off'||v==='silent')return 'None';
 if(v.includes('sizzle'))return 'Sizzle';
 if(v.includes('shaker'))return 'Shaker';
 if(v.includes('hat')||v.includes('hihat'))return 'Hat';
 if(v.includes('snare'))return 'Snare';
 if(v.includes('clap'))return 'Clap';
 if(v.includes('sidestick')||v==='stick')return 'Stick';
 if(v.includes('hammer'))return 'Hammer';
 if(v.includes('chuck'))return 'Chuck';
 if(v.includes('square')||v.includes('chroma'))return 'Tone';
 if(v.includes('slowdown'))return 'Down';
 if(v.includes('speedup'))return 'Up';
 if(v.includes('kick'))return 'Kick';
 return 'Kick';
}
function synth(cat,ctx){
 const hz=ctx.sampleRate;
 const durations={Kick:.20,Snare:.19,Shaker:.19,Hat:.115,Sizzle:.28,Chuck:.16,Clap:.20,Stick:.095,Hammer:.23,Tone:.18,Down:.27,Up:.27};
 const dur=durations[cat]||.16,n=Math.max(1,Math.round(dur*hz));
 const buffer=ctx.createBuffer(1,n,hz),out=buffer.getChannelData(0);
 let seed=123456789,pink=0,phase=0,prevNoise=0;
 function noise(){seed=(1664525*seed+1013904223)>>>0;return (seed/2147483648)-1;}
 for(let i=0;i<n;i++){
  const t=i/hz,p=t/dur,z=noise();
  pink=pink*.84+z*.16;
  const high=z-prevNoise*.82;prevNoise=z;
  let v=0;
  if(cat==='Kick'||cat==='Hammer'||cat==='Chuck'){
   const kick=cat==='Chuck'?140:cat==='Hammer'?84:116;
   const target=cat==='Chuck'?62:cat==='Hammer'?43:45;
   const freq=target+(kick-target)*Math.exp(-t*29);
   phase+=Math.PI*2*freq/hz;
   const low=Math.sin(phase)*Math.exp(-t*(cat==='Hammer'?23:27));
   const click=high*Math.exp(-t*230)*.16;
   v=low*.92+click;
  }else if(cat==='Tone'){
   phase+=Math.PI*2*(480-120*p)/hz;
   v=(Math.sin(phase)*.72+Math.sin(phase*2)*.12)*Math.exp(-t*26);
  }else if(cat==='Up'||cat==='Down'){
   const frequency=cat==='Up'?(250+1250*p):(1250-1000*p);
   phase+=Math.PI*2*frequency/hz;
   v=(Math.sin(phase)+Math.sin(phase*1.9)*.15)*Math.exp(-t*12);
  }else if(cat==='Stick'){
   phase+=Math.PI*2*340/hz;
   v=(.65*high+.28*Math.sin(phase))*Math.exp(-t*92);
  }else if(cat==='Hat'){
   v=(high*.73+z*.16)*Math.exp(-t*58);
  }else if(cat==='Shaker'){
   const rough=Math.sin(t*400)>-.32?1:.24;
   v=high*Math.exp(-t*22)*rough*.82;
  }else if(cat==='Sizzle'){
   v=(high*.8+pink*.18)*Math.exp(-t*12)*(.8+.2*Math.cos(2*Math.PI*37*t));
  }else if(cat==='Snare'){
   phase+=Math.PI*2*175/hz;
   v=(.75*high+.26*Math.sin(phase))*Math.exp(-t*30);
  }else if(cat==='Clap'){
   const clap=Math.exp(-t*47)+.7*Math.exp(-Math.abs(t-.018)*76)+.35*Math.exp(-Math.abs(t-.037)*70);
   v=(high*.86+pink*.1)*clap*.69;
  }
  out[i]=clamp(v,-1,1);
 }
 return buffer;
}
function initContext(){
 if(ENGINE.context)return ENGINE.context;
 const AudioCtx=window.AudioContext||window.webkitAudioContext;
 if(!AudioCtx)return null;
 try{
  const ctx=new AudioCtx({latencyHint:'interactive'});
  ENGINE.context=ctx;
  // Prebuild all common timbres and reuse WebAudio AudioBuffers for each tap.
  for(const name of ['Kick','Snare','Shaker','Hat','Sizzle','Chuck','Clap','Stick','Hammer','Tone','Down','Up']){
   ENGINE.buffers.set(name,synth(name,ctx));
  }
  ENGINE.ready=true;
  if($('hitsoundState'))$('hitsoundState').textContent='준비됨';
  return ctx;
 }catch(e){
  if($('hitsoundState'))$('hitsoundState').textContent='지원하지 않는 브라우저';
  console.warn('hitsound engine unavailable',e);
  return null;
 }
}
function unlock(){
 const ctx=initContext();
 if(ctx?.state==='suspended')ctx.resume().catch(()=>{});
}
function stateAt(floor){
 const doc=window.ADOFAIChartPort?.state?.doc;
 let name=normalizedType(doc?.settings?.hitsound||ENGINE.preset);
 let volume=clamp(Number(doc?.settings?.hitsoundVolume??ENGINE.volume),0,100);
 if(!Number.isFinite(volume))volume=100;
 // Sorted SetHitsound event states. One event can change both parameters.
 let lo=0,hi=ENGINE.eventFloor.length;
 while(lo<hi){const mid=(lo+hi)>>>1;if(ENGINE.eventFloor[mid]<=floor)lo=mid+1;else hi=mid;}
 const at=lo-1;
 if(at>=0){name=ENGINE.eventType[at];volume=ENGINE.eventVolume[at];}
 return {name,volume};
}
function play(floor,opts={}){
 if(!ENGINE.enabled)return false;
 const setting=stateAt(Number.isInteger(floor)?floor:0);
 if(sampleCategory(setting.name)==='None'||setting.volume<=0)return false;
 const ctx=initContext();if(!ctx)return false;
 const playback=()=>{
  const buf=ENGINE.buffers.get(sampleCategory(setting.name))||ENGINE.buffers.get('Kick');
  const gain=ctx.createGain();
  gain.gain.value=clamp(setting.volume/100,0,1)*.48;
  const source=ctx.createBufferSource();source.buffer=buf;
  source.connect(gain).connect(ctx.destination);
  source.start(ctx.currentTime);
  source.onended=()=>{source.disconnect();gain.disconnect();};
  ENGINE.lastPlayed={floor,name:setting.name,volume:setting.volume,time:ctx.currentTime};
 };
 if(ctx.state==='running'){playback();return true;}
 // First user gesture can unblock a suspended context. Do not recreate buffers.
 ctx.resume().then(playback).catch(()=>{});
 return true;
}
function sync(doc){
 if(!doc||!doc.settings)return;
 ENGINE.preset=normalizedType(doc.settings.hitsound||'Kick');
 const v=Number(doc.settings.hitsoundVolume??100);ENGINE.volume=Number.isFinite(v)?clamp(v,0,100):100;
 let name=ENGINE.preset,volume=ENGINE.volume;
 const events=(doc.actions||[]).map((event,index)=>({...event,__order:index}))
   .filter(e=>e.eventType==='SetHitsound'&&Number.isInteger(e.floor)&&e.active!==false&&(!e.gameSound||e.gameSound==='Hitsound'))
   .sort((a,b)=>a.floor-b.floor||a.__order-b.__order);
 ENGINE.eventFloor=[];ENGINE.eventType=[];ENGINE.eventVolume=[];
 for(const e of events){
  if(e.hitsound!=null)name=normalizedType(e.hitsound);
  const v=Number(e.hitsoundVolume);
  if(e.hitsoundVolume!=null&&Number.isFinite(v))volume=clamp(v,0,100);
  ENGINE.eventFloor.push(e.floor);
  ENGINE.eventType.push(name);
  ENGINE.eventVolume.push(volume);
 }
 if($('hitsoundType'))$('hitsoundType').textContent=ENGINE.preset;
 if($('hitsoundVol'))$('hitsoundVol').textContent=fmt(ENGINE.volume)+'%';
 updateCurrent();
 const song=$('chartAudio');
 if(song){
  const musicVolume=Number(doc.settings.volume??100);
  if(Number.isFinite(musicVolume))song.volume=clamp(musicVolume/100,0,1);
 }
}
function fmt(n){return Number.isInteger(n)?String(n):Number(n).toFixed(1);}
function updateCurrent(){
 const floor=window.ADOFAIArchivePlayer?.state?.playing?window.ADOFAIArchivePlayer.state.playerIndex:window.ADOFAIChartPort?.state?.selected??0;
 const current=stateAt(floor);
 if($('hitsoundCurrent'))$('hitsoundCurrent').textContent='현재 '+current.name+' · '+fmt(current.volume)+'%';
}
function boot(){
 const transport=document.querySelector('.pc-transport');if(!transport)return;
 const box=document.createElement('div');
 box.id='hitsoundBar';
 box.innerHTML='<span><strong>히트사운드</strong> <span id="hitsoundType">Kick</span> <span id="hitsoundVol">100%</span></span>'+
  '<span id="hitsoundCurrent">현재 Kick · 100%</span>'+
  '<label><input type="checkbox" id="hitsoundEnabled" checked> 타일 판정 시 재생</label>'+
  '<button id="hitsoundTry" type="button">소리 테스트</button>'+
  '<span class="hs-status" id="hitsoundState">첫 입력 시 활성화</span>'+
  '<small>차트의 hitsound / hitsoundVolume / SetHitsound 적용 · Web Audio 근사 음색</small>';
 transport.insertAdjacentElement('afterend',box);
 const style=document.createElement('style');
 style.textContent='#hitsoundBar{background:#152431;border-bottom:1px solid #415565;padding:8px 14px;display:flex;align-items:center;gap:9px 14px;flex-wrap:wrap;font-size:12px;color:#cfe0f0}'+
 '#hitsoundBar>span:first-child{font-size:13px;color:#fff0bb}#hitsoundBar strong{font-size:13px}#hitsoundBar label{display:flex;align-items:center;gap:5px}'+
 '#hitsoundBar button{border:1px solid #698297;background:#31485d;color:#edf7ff;border-radius:4px;padding:6px 11px;cursor:pointer}'+
 '#hitsoundBar small{color:#95b0c6}#hitsoundState{color:#8cb7c8;font-size:11px}';
 document.head.appendChild(style);
 $('hitsoundTry').onclick=()=>play(window.ADOFAIChartPort?.state?.selected||0);
 $('hitsoundEnabled').onchange=e=>{ENGINE.enabled=e.target.checked;};
 document.addEventListener('pointerdown',unlock,{capture:true,passive:true});
 document.addEventListener('keydown',unlock,{capture:true,passive:true});
 document.addEventListener('touchstart',unlock,{capture:true,passive:true});
 window.addEventListener('adofai:timing-updated',()=>{
  const doc=window.ADOFAIChartPort?.state?.doc;if(doc)sync(doc);
 });
 setInterval(()=>{if(window.ADOFAIChartPort?.state?.doc)updateCurrent();},250);
 const doc=window.ADOFAIChartPort?.state?.doc;if(doc)sync(doc);
}
window.ADOFAIHitsound={init:initContext,unlock,play,sync,stateAt,get state(){return ENGINE;}};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();
