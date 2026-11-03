/* One upstream Web ADOFAI frame and an optional editor drawer.
   Original game code is served by its author. No 2nd level-search iframe. */
(()=>{'use strict';
 const $=id=>document.getElementById(id);
 const ORIGIN='https://web-adofai.impl1113.dev';
 const api=()=>window.ADOFAINativeStudio;
 const isNative=new URLSearchParams(location.search).get('native')==='1';
 function status(msg,kind='error'){const e=$('status');if(e){e.textContent=msg;e.className='status '+kind}}
 function openDrawer(){$('tools').classList.remove('collapsed');$('editorScrim').hidden=false;$('toggleDrawer').setAttribute('aria-expanded','true');}
 function closeDrawer(){$('tools').classList.add('collapsed');$('editorScrim').hidden=true;$('toggleDrawer').setAttribute('aria-expanded','false');}
 function toggleDrawer(){
  if(!$('tools').classList.contains('collapsed')){closeDrawer();return}
  if(api()?.state.doc&&api().state.mode!=='editor'){
   void api().selectMode('editor').then(()=>openDrawer()).catch(e=>status('편집 화면 오류: '+e.message));
  }else openDrawer();
 }
 function showLevels(){
  if(api()?.state)api().state.mode='search';
  $('engine').src=ORIGIN+'/levels';
  document.body.dataset.pageView='levels';$('showBrowser').setAttribute('aria-pressed','true');
  $('microHint').hidden=true;closeDrawer();
 }
 function refresh(){
  const s=api()?.state;if(!s)return;
  $('quickSave').disabled=!s.doc;
  if(s.doc&&s.mode==='editor'){
   document.body.dataset.pageView='editor';$('microHint').hidden=false;
   $('mapSubheading').textContent=s.filename||'커스텀 차트';
  }
 }
 function activateEditor(){
  document.body.dataset.pageView='editor';$('microHint').hidden=false;
  $('showBrowser').setAttribute('aria-pressed','false');
  refresh();closeDrawer();
 }
 function boot(){
  if(!api()||!$('engine'))return;
  document.body.dataset.native=String(isNative);document.body.dataset.pageView='levels';
  $('showBrowser').textContent='🎮 레벨';$('toggleDrawer').textContent='✎ 편집';
  $('quickOpen').textContent='📂 열기';$('quickSave').textContent='💾 저장';
  // Remove redundant search container and its second iframe.
  $('browserPanel')?.remove();
  const title=document.createElement('div');title.id='studioDrawerHeading';
  title.innerHTML='<strong>커스텀 레벨 편집</strong><button id="cleanEditorClose" type="button" aria-label="편집 닫기">×</button>';
  $('tools').prepend(title);
  const backdrop=document.createElement('div');backdrop.id='editorScrim';backdrop.hidden=true;
  document.querySelector('.app').append(backdrop);
  const hint=document.createElement('div');hint.id='microHint';hint.hidden=true;
  hint.textContent='원본 Web ADOFAI 차트 렌더러 · 타일 선택 / 편집 / 저장';
  document.querySelector('.stage').append(hint);
  $('cleanEditorClose').onclick=closeDrawer;backdrop.onclick=closeDrawer;
  $('toggleDrawer').onclick=toggleDrawer;
  $('showBrowser').onclick=showLevels;
  $('quickOpen').onclick=()=>$('picker').click();
  $('quickSave').onclick=()=>void window.ADOFAINativeStudioV2?.exportZip();
  const u=window.ADOFAIUnifiedUI;
  if(u){u.chartActivated=activateEditor;u.openBrowser=showLevels;u.closeBrowser=()=>{};u.chartLoaded=()=>{u.refresh?.();u.centerSelected?.();refresh()};}
  const info=$('info');if(info)new MutationObserver(refresh).observe(info,{childList:true,subtree:true,characterData:true});
  window.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('tools').classList.contains('collapsed'))closeDrawer()});
  closeDrawer();refresh();
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();