/* Unified one-screen workspace for Web ADOFAI.
   No copied upstream engine or claim of playable custom engine in public build. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
const UPSTREAM='https://web-adofai.impl1113.dev';
const api=()=>window.ADOFAINativeStudio;
let ready=false,lastFocused=-1,timer=0;
function status(text,kind='error'){const x=$('status');if(x){x.textContent=text;x.className='status '+kind}}
function refresh(){
 const s=api()?.state;if(!s)return;
 const n=s.doc?api().angles().length+1:0,loaded=!!s.doc;
 $('mapSubheading').textContent=loaded?s.filename+' · '+n+'타일':'차트를 열면 여기에서 곧바로 편집합니다';
 $('floorReadout').textContent=loaded?s.selected+'/'+(n-1):'— / —';
 for(const [id,disabled] of [['stepBack',!loaded||s.selected<=0],['stepForward',!loaded||s.selected>=n-1],['centerFloor',!loaded],['quickSave',!loaded],['applyEngine',!loaded],['playEngine',!loaded]])$(id).disabled=disabled;
 if(!loaded)lastFocused=-1;
}
function focusFloor(floor){
 if(!api()?.state.doc)return;
 const n=Math.max(0,Math.min(api().angles().length,Number(floor)));
 if(!Number.isInteger(n)||n===lastFocused)return;
 lastFocused=n;
 if(new URLSearchParams(location.search).get('native')==='1')return;
 $('engine').contentWindow?.postMessage({protocol:'web-adofai.chart',version:1,type:'chart.focus',floorIndex:n},UPSTREAM);
}
function centerSelected(force=false){if(!api()?.state.doc)return;if(force)lastFocused=-1;focusFloor(api().state.selected)}
function chartLoaded(){lastFocused=-1;setTimeout(()=>centerSelected(),250);refresh()}
function floorChanged(){refresh();clearTimeout(timer);timer=setTimeout(()=>centerSelected(),70)}
function openBrowser(){
 if(!ready)return;
 const panel=$('browserPanel');
 panel.hidden=false;$('showBrowser').setAttribute('aria-pressed','true');
 const frame=$('library');if(!frame.getAttribute('src'))frame.src=UPSTREAM+'/levels';
}
function closeBrowser(){
 const panel=$('browserPanel');if(!panel)return;
 panel.hidden=true;panel.classList.remove('expanded');
 $('showBrowser').setAttribute('aria-pressed','false');
 $('browserExpand').setAttribute('aria-pressed','false');$('browserExpand').textContent='▣ 크게';
}
function chartActivated(){refresh();closeBrowser();$('siteNotice').hidden=true}
function toggleBrowser(){if($('browserPanel').hidden)openBrowser();else closeBrowser()}
function importDropped(files){
 if(!files?.length)return;
 const extras=window.ADOFAINativeStudioV2;if(!extras){status('파일 관리 모듈이 없습니다.');return}
 if(files.length>1)void extras.importDirectory(Array.from(files)).catch(e=>status('파일 불러오기 실패: '+e.message));
 else void extras.openAny(files[0]);
}
function boot(){
 if(!api())return;
 ready=true;
 $('showBrowser').onclick=toggleBrowser;
 $('closeBrowser').onclick=closeBrowser;
 $('browserExpand').onclick=()=>{
  const flag=$('browserPanel').classList.toggle('expanded');
  $('browserExpand').setAttribute('aria-pressed',String(flag));
  $('browserExpand').textContent=flag?'▤ 작게':'▣ 크게';
 };
 $('quickOpen').onclick=()=>$('picker').click();
 $('quickSave').onclick=()=>void window.ADOFAINativeStudioV2?.exportZip();
 $('stepBack').onclick=()=>{api().selectFloor(api().state.selected-1);floorChanged()};
 $('stepForward').onclick=()=>{api().selectFloor(api().state.selected+1);floorChanged()};
 $('centerFloor').onclick=()=>centerSelected(true);
 $('closeNotice').onclick=()=>$('siteNotice').hidden=true;
 const stage=document.querySelector('.stage');
 stage.addEventListener('dragover',e=>{if(e.dataTransfer?.types?.includes('Files'))e.preventDefault()});
 stage.addEventListener('drop',e=>{if(e.dataTransfer?.files?.length){e.preventDefault();importDropped(e.dataTransfer.files)}});
 const info=$('info');new MutationObserver(refresh).observe(info,{childList:true,characterData:true,subtree:true});
 window.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&!$('browserPanel').hidden){closeBrowser();e.preventDefault()}
  if(e.key==='F8'&&!e.metaKey&&!e.ctrlKey){toggleBrowser();e.preventDefault()}
 });
 if(matchMedia('(max-width:760px)').matches){
  $('tools').classList.add('collapsed');$('toggleDrawer').setAttribute('aria-expanded','false');
 }
 refresh();
}
window.ADOFAIUnifiedUI={openBrowser,closeBrowser,chartActivated,chartLoaded,floorChanged,refresh,focusFloor,centerSelected};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();
