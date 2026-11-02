/* Original PC scene-linked editor: ZIP chart + sibling assets + rhythm practice.
 * Reads user-provided data locally. Does not upload media or claim Unity engine parity.
 * JSZip v3.10.1 (MIT) for ZIP parsing and lossless archive round trips.
 */
(function(){
'use strict';
const $=id=>document.getElementById(id);
const api=()=>window.ADOFAIChartPort;
const ctx={zip:null,entry:'',zipName:'',songUrl:'',imageUrl:'',chartName:'',timings:[],playerIndex:0,judged:new Set(),playing:false,frame:0,lastActive:-1,good:0,miss:0,estimate:[],inputFile:null};
const safe = value => String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const norm=a=>((a%360)+360)%360;
function msg(text,isError=false){const el=$('archiveStatus');if(el){el.textContent=text;el.dataset.error=isError?'true':'false'} const chart=$('chartStatus');if(chart)chart.textContent=text;}
function disposeMedia(){for(const key of ['songUrl','imageUrl'])if(ctx[key]){URL.revokeObjectURL(ctx[key]);ctx[key]=''} const a=$('chartAudio');if(a){a.pause();a.removeAttribute('src');a.load()};$('chartBackdrop').style.backgroundImage='';}
function pickChart(entries){
 const charts=entries.filter(n=>/\.adofai$/i.test(n)&&!/\/__MACOSX\//i.test('/'+n));
 if(!charts.length)throw Error('ZIP 안에 .adofai 차트가 없습니다.');
 const ranking = n=>{const base=n.split('/').pop();let score=0;if(/backup|back[-_ ]?up|자동저장|autosave/i.test(n))score+=100;if(/\/\./.test(n))score+=40;if(base.toLowerCase()==='level.adofai')score-=10;score+=n.split('/').length;return score};
 const sorted=charts.sort((a,b)=>ranking(a)-ranking(b));
 if(sorted.length===1)return sorted[0];
 // Intentionally choose the primary chart ahead of a backup without a blocking prompt.
 return sorted[0];
}
function resolveAsset(filename){
 if(!ctx.zip||!filename)return null;
 const path=filename.replace(/\\/g,'/');if(path.startsWith('/')||path.includes('../'))return null;
 const dir=ctx.entry.slice(0,ctx.entry.lastIndexOf('/')+1);
 const all=Object.keys(ctx.zip.files).filter(n=>!ctx.zip.files[n].dir);
 const prefer=[dir+path,path];
 for(const p of prefer){const exact=all.find(n=>n===p)||all.find(n=>n.toLowerCase()===p.toLowerCase());if(exact)return exact;}
 const base=path.split('/').pop().toLowerCase();return all.find(n=>n.split('/').pop().toLowerCase()===base)||null;
}
async function mediaAssets(){
 disposeMedia();let doc=api().saveObject();let song=resolveAsset(doc.settings?.songFilename||'');
 if(!song&&ctx.zip)song=resolveAsset('audio.ogg');
 if(song){const file=ctx.zip.file(song);if(file){const b=await file.async('blob');if(b.size>80*1024*1024){msg('음원 파일이 너무 커서 메모리에 올리지 않았습니다.',true)}else {ctx.songUrl=URL.createObjectURL(new Blob([b],{type:/\.ogg$/i.test(song)?'audio/ogg':'audio/mpeg'}));$('chartAudio').src=ctx.songUrl;$('chartAudioName').textContent=song.split('/').pop();}}}
 const background=resolveAsset(doc.settings?.bgImage)||resolveAsset(doc.settings?.previewImage);
 if(background&&/\.(jpe?g|png|webp|gif)$/i.test(background)){
  const bytes=await ctx.zip.file(background).async('blob');if(bytes.size<20*1024*1024){ctx.imageUrl=URL.createObjectURL(bytes);$('chartBackdrop').style.backgroundImage='url("'+ctx.imageUrl+'")';}
 }
 $('archiveAssets').textContent=(ctx.zip?Object.keys(ctx.zip.files).filter(n=>!ctx.zip.files[n].dir).length:0)+'개 파일 · 음악 '+(song?'찾음':'없음')+' · 이미지 '+(background?'찾음':'없음');
}
async function openZip(file){
 try{
  if(!window.JSZip)throw Error('ZIP 라이브러리를 불러오지 못했습니다.');
  msg('ZIP 압축을 읽는 중…');stopPlay();
  const arr=await file.arrayBuffer();
  if(arr.byteLength>250*1024*1024)throw Error('250MB를 넘는 ZIP은 브라우저 메모리 제한으로 지원하지 않습니다.');
  const zip=await JSZip.loadAsync(arr,{checkCRC32:true});
  const primary=pickChart(Object.keys(zip.files));
  const s=await zip.file(primary).async('string');
  const doc=api().parseChart(s);
  const angles=Array.isArray(doc.angleData)?doc.angleData.length:doc.pathData.length;
  if(angles>50000)throw Error('타일 5만 개 초과 차트는 아직 열 수 없습니다.');
  // Only change port state after CRC/JSON validation. Failed ZIPs do not replace a chart.
  ctx.zip=zip;ctx.entry=primary;ctx.zipName=file.name;ctx.chartName=primary.split('/').pop();ctx.inputFile=file;
  api().loadObject(doc,ctx.chartName);
  await mediaAssets();updateTiming();
  $('chartZipSave').disabled=false;$('chartSave').textContent='ZIP 저장';$('chartSaveAs').textContent='차트만 저장';
  msg(file.name+' 불러오기 성공 · '+primary+' · '+(angles+1)+'개 타일 · 음악 '+(ctx.songUrl?'연결됨':'없음'));
  $('chartStageBody')?.scrollIntoView({block:'nearest'});
 }catch(err){msg('ZIP 열기 실패: '+err.message,true);console.error('ZIP failed',err);}
}
function flushZipMode(){ctx.zip=null;ctx.entry='';ctx.zipName='';ctx.chartName='';ctx.inputFile=null;stopPlay();disposeMedia();$('chartZipSave').disabled=true;$('chartSave').textContent='저장';$('chartSaveAs').textContent='다른 이름 저장';$('archiveAssets').textContent='';}
function interceptInput(e){
 const file=e.target?.files?.[0];if(!file)return;
 if(/\.(zip|adofai\.zip)$/i.test(file.name)){
   e.stopImmediatePropagation();e.preventDefault();
   openZip(file);e.target.value='';return;
 }
 // Plain .adofai should still flow into the existing editor handler.
 flushZipMode();const tick=()=>{if(api().saveObject()){updateTiming();msg(file.name+' 차트를 열었습니다. 음악 파일은 따로 선택할 수 있습니다.')}};
 setTimeout(tick,30);
}
function download(blob,name){const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),20000)}
async function saveZip(chooseName=false){
 if(!ctx.zip||!ctx.entry){msg('먼저 ZIP 파일을 열어주세요.',true);return}
 try{
  msg('음악·백업·이미지를 보존하면서 ZIP을 저장하는 중…');
  const doc=api().saveObject();if(!doc)throw Error('열린 차트가 없습니다.');
  const copied=await JSZip.loadAsync(await ctx.inputFile.arrayBuffer());
  copied.file(ctx.entry,JSON.stringify(doc,null,2)+'\n');
  const compressed=await copied.generateAsync({type:'blob',compression:'DEFLATE',compressionOptions:{level:6}});
  let name=(ctx.zipName||'level.zip').replace(/\.zip$/i,'-edited.zip');
  if(chooseName){const requested=prompt('저장할 ZIP 파일명',name);if(requested===null)return;name=requested.trim()||name;if(!/\.zip$/i.test(name))name+='.zip';}
  download(compressed,name);msg('ZIP 생성 완료: '+name+' · 원본 음악·백업 등 유지');
 }catch(e){msg('ZIP 저장 실패: '+e.message,true);console.error(e)}
}
function onSaveCapture(e){if(!ctx.zip)return;e.preventDefault();e.stopImmediatePropagation();saveZip(false)}
function onSaveAsCapture(e){if(!ctx.zip)return;e.preventDefault();e.stopImmediatePropagation();const doc=api().saveObject();if(!doc)return;download(new Blob([JSON.stringify(doc,null,2)+'\n'],{type:'application/json'}),ctx.chartName||'level.adofai');msg('.adofai 단독 파일 저장 완료');}
async function loadAudioFile(file){if(!file)return;stopPlay();if(ctx.songUrl)URL.revokeObjectURL(ctx.songUrl);ctx.songUrl=URL.createObjectURL(file);$('chartAudio').src=ctx.songUrl;$('chartAudioName').textContent=file.name;updateTiming();msg('로컬 음악 연결됨: '+file.name)}
function updateTiming(){
 const doc=api().saveObject();if(!doc){ctx.timings=[];return}
 const angles=api().state.angles||[],actions=Array.isArray(doc.actions)?doc.actions:[];
 const groups=new Map();for(const e of actions){if(!Number.isInteger(e?.floor))continue;const group=groups.get(e.floor)||[];group.push(e);groups.set(e.floor,group)}
 let bpm=Math.max(1,Number(doc.settings?.bpm)||120);
 let t=(Number(doc.settings?.offset)||0)/1000;
 const timings=[t];let twirl=false;let previous=0;
 const warnings=new Set();
 for(let floor=0;floor<=angles.length;floor++){
  const evs=groups.get(floor)||[];
  for(const e of evs){
   if(e.eventType==='Twirl')twirl=!twirl;
   if(e.eventType==='SetSpeed'){
    if(e.speedType==='Bpm'&&Number(e.beatsPerMinute)>0)bpm=Number(e.beatsPerMinute);
    if(e.speedType==='Multiplier'&&Number(e.bpmMultiplier)>0)bpm*=Number(e.bpmMultiplier);
   }
   if(e.eventType==='Pause'&&Number(e.duration)>0)t+=Number(e.duration)*60/bpm;
   if(e.eventType==='Hold')warnings.add('Hold');
  }
  if(floor>=angles.length)break;
  const next=angles[floor];let travel=180;
  if(next===999){travel=0;}
  else if(floor>0){travel=norm(180+previous-next);if(travel<.001)travel=360;}
  if(next!==999)previous=next;
  if(twirl)travel=(360-travel)%360||360;
  t+=travel/180*60/Math.max(bpm,1);
  timings.push(t);
 }
 ctx.timings=timings;
 ctx.estimate=Array.from(warnings);
 const warn=ctx.estimate.length?' · 미구현 판정: '+ctx.estimate.join(', '):'';
 $('chartTimingInfo').textContent='연습 타이밍 추정 · '+timings.length+' 타일 · 기본 '+doc.settings?.bpm+' BPM · offset '+(doc.settings?.offset||0)+'ms'+warn;
}
function currentIndex(time){let lo=0,hi=ctx.timings.length;while(lo<hi){const mid=(lo+hi)>>1;if(ctx.timings[mid]<=time)lo=mid+1;else hi=mid}return Math.max(0,lo-1)}
function updateLive(){if(!ctx.playing)return;const time=$('chartAudio').currentTime;const floor=currentIndex(time);ctx.playerIndex=floor;
 if(floor!==ctx.lastActive){const old=$('chartSvg').querySelector('.chart-floor.live');if(old)old.classList.remove('live');const cur=$('chartSvg').querySelector('[data-floor="'+floor+'"]');if(cur)cur.classList.add('live');ctx.lastActive=floor;}
 $('chartPlayProgress').textContent='진행 '+floor+'/'+Math.max(0,ctx.timings.length-1)+' · '+time.toFixed(2)+'s · 성공 '+ctx.good+' / 놓침 '+ctx.miss;
 if($('chartAudio').ended){stopPlay();return}ctx.frame=requestAnimationFrame(updateLive)
}
async function startPlay(){
 try{
  if(!api().saveObject()){msg('차트를 먼저 열어주세요.',true);return}
  if(!$('chartAudio').src){msg('음악을 선택하거나 ZIP에 포함된 음원을 사용하세요.',true);return}
  updateTiming();if(ctx.timings.length===0)return;
  const floor=api().state.selected||0;
  const seek=Math.max(0,ctx.timings[Math.min(floor,ctx.timings.length-1)]-.5);
  if(!ctx.playing){$('chartAudio').currentTime=seek;ctx.judged.clear();ctx.good=0;ctx.miss=0}
  await $('chartAudio').play();ctx.playing=true;cancelAnimationFrame(ctx.frame);ctx.lastActive=-1;updateLive();
  msg('연습 플레이 시작 · 스페이스/터치로 타이밍 입력 · Unity 원본 판정과 차이가 있을 수 있음');
 }catch(e){msg('연습 플레이 실패: '+e.message,true)}
}
function stopPlay(){ctx.playing=false;cancelAnimationFrame(ctx.frame);$('chartAudio')?.pause();const old=$('chartSvg')?.querySelector('.chart-floor.live');if(old)old.classList.remove('live');ctx.lastActive=-1;}
function tap(){if(!ctx.playing||!ctx.timings.length)return;const t=$('chartAudio').currentTime;
 const next=Math.min(ctx.timings.length-1,currentIndex(t)+1),prior=Math.max(1,next-1);
 const candidates=[next,prior].filter(i=>!ctx.judged.has(i));candidates.sort((a,b)=>Math.abs(ctx.timings[a]-t)-Math.abs(ctx.timings[b]-t));
 const hit=candidates[0];if(hit===undefined)return;const err=Math.abs(ctx.timings[hit]-t);if(err<=.16){ctx.good++;ctx.judged.add(hit);$('chartTapButton').dataset.grade=err<=.055?'perfect':'good';$('chartGrade').textContent=err<=.055?'PERFECT':'GOOD';}else{$('chartGrade').textContent='MISS';ctx.miss++;$('chartTapButton').dataset.grade='miss';}
 $('chartGrade').style.opacity='1';setTimeout(()=>$('chartGrade').style.opacity='.35',250);
}
function createUi(){const port=$('chartPort');if(!port)return;
 // This is a browser compatibility layer, not a claim that Unity PC native C# is running.
 const toolbar=port.querySelector('.chart-toolbar');
 const input=$('chartFile');input.accept='.adofai,.json,.zip,application/zip,application/json';
 input.addEventListener('change',interceptInput,true);
 $('chartSave').addEventListener('click',onSaveCapture,true);
 $('chartSaveAs').addEventListener('click',onSaveAsCapture,true);
 const zipButton=document.createElement('button');zipButton.id='chartZipSave';zipButton.textContent='ZIP 전체 저장';zipButton.disabled=true;zipButton.onclick=()=>saveZip(true);$('chartSaveAs').after(zipButton);
 const media=document.createElement('div');media.className='pc-media';media.innerHTML=`<div class="pc-menu"><span>FILE</span><span>EDIT</span><span>LEVEL</span><span>EVENTS</span><span>DECORATIONS</span><span>SETTINGS</span><small>PC <code>scnEditor</code> hierarchy linked</small></div><div class="pc-transport"><div class="pc-transport-left"><button id="chartPlay" type="button">▶ 연습 플레이</button><button id="chartStop" type="button">■ 정지</button><button id="chartTapButton" type="button">TAP / SPACE</button><span id="chartGrade"></span></div><div class="pc-transport-right"><audio id="chartAudio" controls preload="metadata"></audio><label class="pc-load-song">음악 선택<input id="chartLocalSong" type="file" accept="audio/*,.ogg,.mp3,.wav" hidden></label><span id="chartAudioName">음원 없음</span></div></div><div class="pc-stats"><span id="chartPlayProgress">ZIP을 열면 음악과 차트가 자동 연결됩니다.</span><span id="chartTimingInfo">연습 플레이는 원본 Unity 판정이 아닌 브라우저 타이밍 추정입니다.</span><span id="archiveAssets"></span></div><div id="archiveStatus" role="status" aria-live="polite">.adofai 또는 .zip 파일을 열어주세요.</div>`;
 toolbar.insertAdjacentElement('afterend',media);
 const body=$('chartWorkspace');body.id='chartStageBody';const bg=document.createElement('div');bg.id='chartBackdrop';$('chartSvg').parentNode.insertBefore(bg,$('chartSvg'));
 $('chartPlay').onclick=startPlay;$('chartStop').onclick=stopPlay;$('chartTapButton').onclick=tap;$('chartLocalSong').onchange=e=>loadAudioFile(e.target.files[0]);
 $('chartAudio').addEventListener('ended',stopPlay);
 $('chartAudio').addEventListener('seeked',()=>{ctx.lastActive=-1});
 window.addEventListener('keydown',e=>{if(e.repeat||!ctx.playing)return;const target=e.target?.tagName||'';if(/INPUT|TEXTAREA|SELECT/.test(target)||e.target?.isContentEditable)return;if(e.code==='Space'||e.code==='Enter'){e.preventDefault();tap()}});
 const css=document.createElement('style');css.textContent=`
 .top{background:#14191f;border-color:#333d46}.chart-port{background:#171d23!important;border-top:1px solid #37414a!important;border-bottom-color:#34424b!important}
 .chart-toolbar{background:#1e252c!important;padding:12px 14px!important;margin:0 -15px}.chart-brand{font-size:13px!important;color:#ebbd78!important}
 .chart-buttons button,.chart-tabs button,.chart-angle-bar button,.pc-transport button,.pc-load-song{border-radius:3px!important;background:#333e48!important;border:1px solid #56636d!important;color:#f2f3f6!important;font-weight:600;cursor:pointer}
 .chart-port button:hover:not(:disabled),.pc-transport button:hover{background:#4d5e6e!important}
 #chartWorkspace{padding:8px 0!important}.chart-stage,.chart-side{background:#1c252d!important;border-color:#49545f!important}.chart-stage-head{background:#273440!important}
 #chartSvg{background:radial-gradient(circle at 55% 46%,#1d2832,#0b1019)!important;position:relative;z-index:1;touch-action:pan-x pan-y}
 .chart-stage{position:relative}.pc-media{background:#1a212a;margin:0 -15px;border-bottom:1px solid #55606a}
 .pc-menu{display:flex;gap:22px;padding:9px 16px;border-bottom:1px solid #3d4853;font-size:11px;font-weight:700;color:#dee4ea;flex-wrap:wrap;align-items:center}.pc-menu small{margin-left:auto;color:#85949e;font-weight:400}
 .pc-transport{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;padding:10px 14px;gap:10px}.pc-transport-left,.pc-transport-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
 #chartAudio{height:32px;width:215px;max-width:80vw}#chartAudioName{max-width:150px;text-overflow:ellipsis;overflow:hidden;white-space:nowrap;font-size:11px;color:#acbac5}.pc-load-song{padding:7px 9px;font-size:12px;display:inline-block}
 .pc-stats{padding:0 15px 8px;display:flex;gap:20px;font-size:11px;color:#a9b5bf;flex-wrap:wrap}
 #archiveStatus{border-top:1px solid #353f49;padding:7px 15px;font-size:12px;color:#bcd4e5}#archiveStatus[data-error=true]{color:#ffad9d}
 #chartGrade{font-weight:800;color:#f8c97c;min-width:65px}#chartTapButton[data-grade=perfect]{border-color:#f0d36f!important}#chartTapButton[data-grade=good]{border-color:#8ad7b7!important}#chartTapButton[data-grade=miss]{border-color:#d87d7d!important}
 .chart-floor.live rect{stroke:#ffda6d;stroke-width:5px;filter:drop-shadow(0 0 8px #ffac00)}
 #chartBackdrop{position:absolute;inset:49px 0 55px;background-size:cover;background-position:center;opacity:.16;pointer-events:none;z-index:0}
 .chart-side textarea{background:#101921!important}#chartViewer{opacity:.75}
 @media(max-width:740px){.pc-menu{gap:12px}.pc-transport-left button{font-size:11px}.pc-stats{gap:8px}#chartAudio{width:170px}}
 `;document.head.appendChild(css);
}
window.ADOFAIArchivePlayer={get state(){return ctx},loadZip:openZip,saveZip,updateTiming,startPlay,stopPlay,tap};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',createUi);else createUi();
})();
