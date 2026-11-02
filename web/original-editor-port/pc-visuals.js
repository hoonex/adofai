/* PC-scene floor/planet visualization. The original PC level1
   PlanetarySystem/PlanetRed/PlanetBlue/Pivot hierarchy and original
   sharedassets1 tile_white sprite were inspected and used as reference.
   Geometry and timing are browser implementations, not executed Unity code. */
(function(){
'use strict';
const UNIT=54, el=id=>document.getElementById(id);
const S={api:null,svg:null,planets:null,writing:false};
const defs='<defs><radialGradient id="fireBall" cx="30%" cy="22%" r="82%"><stop stop-color="#fffcea"/><stop offset=".37" stop-color="#ffad71"/><stop offset=".72" stop-color="#f04439"/><stop offset="1" stop-color="#9b101c"/></radialGradient><radialGradient id="iceBall" cx="30%" cy="22%" r="82%"><stop stop-color="#fff"/><stop offset=".37" stop-color="#a5eeff"/><stop offset=".72" stop-color="#489ee8"/><stop offset="1" stop-color="#1258a4"/></radialGradient><pattern id="pcTileTexture" width="54" height="54" patternUnits="userSpaceOnUse"><image id="pcTileSprite" width="54" height="54" preserveAspectRatio="none"/></pattern><filter id="planetBlur"><feGaussianBlur stdDeviation="5"/></filter></defs>';
function floorPolygon(x,y){const c=23,r=7;return 'M '+(x-c+r)+' '+(y-c)+' L '+(x+c-r)+' '+(y-c)+' Q '+(x+c)+' '+(y-c)+' '+(x+c)+' '+(y-c+r)+' L '+(x+c)+' '+(y+c-r)+' Q '+(x+c)+' '+(y+c)+' '+(x+c-r)+' '+(y+c)+' L '+(x-c+r)+' '+(y+c)+' Q '+(x-c)+' '+(y+c)+' '+(x-c)+' '+(y+c-r)+' L '+(x-c)+' '+(y-c+r)+' Q '+(x-c)+' '+(y-c)+' '+(x-c+r)+' '+(y-c)+' Z';}
function balls(){
 const ball=(name,color,glow)=>'<g id="'+name+'"><circle r="25" fill="'+glow+'" opacity=".28" filter="url(#planetBlur)"/><circle r="17" fill="url(#'+color+')" stroke="#e9f6ff" stroke-width="2"/><circle cx="-6" cy="-7" r="4.6" fill="#fff" opacity=".75"/></g>';
 return '<g id="pcOrbitingPlanets" pointer-events="none"><line id="pcOrbitLink" x1="0" y1="0" x2="0" y2="0" stroke="#f7f8fb" stroke-width="2" stroke-opacity=".5"/>'+ball('pcFire','fireBall','#f63b3a')+ball('pcIce','iceBall','#35b3ff')+'</g>';
}
function draw(){
 const api=S.api,svg=S.svg,st=api?.state;
 if(!st?.doc||!svg)return;
 const points=st.positions||[];if(!points.length)return;
 S.twirlFloors=(st.doc.actions||[]).filter(e=>e&&e.eventType==='Twirl'&&Number.isInteger(e.floor)).map(e=>e.floor).sort((a,b)=>a-b);
 const selection=st.selected||0,actions=new Set((st.doc.actions||[]).filter(v=>v&&Number.isInteger(v.floor)).map(v=>v.floor)),decos=new Set((st.doc.decorations||[]).filter(v=>v&&Number.isInteger(v.floor)).map(v=>v.floor));
 let from=0,to=points.length;if(points.length>6500){from=Math.max(0,selection-2300);to=Math.min(points.length,selection+2300);}
 let path='';for(let i=from;i<to;i++){const p=points[i];path+=(i===from?'M':'L')+(p.x*UNIT).toFixed(2)+' '+(p.y*UNIT).toFixed(2)+' ';}
 let html=defs+'<g id="pcFloorTrack"><path d="'+path+'" stroke="#0c1825" stroke-width="57" fill="none" stroke-linejoin="round" stroke-linecap="round"/><path d="'+path+'" stroke="#ceddeb" stroke-width="51" fill="none" stroke-linejoin="round" stroke-linecap="round"/><path d="'+path+'" stroke="#526981" stroke-width="47" fill="none" stroke-linejoin="round" stroke-linecap="round"/></g><g id="pcFloorCells">';
 for(let i=from;i<to;i++){
  const p=points[i],x=p.x*UNIT,y=p.y*UNIT,select=i===selection,mid=i>0&&st.angles[i-1]===999;
  const polygon=floorPolygon(x,y);
  html+='<g class="chart-floor'+(select?' selected-floor':'')+'" role="button" tabindex="0" aria-label="타일 '+i+'" data-floor="'+i+'" style="cursor:pointer">'+
    '<path class="pc-tile" d="'+polygon+'" fill="'+(mid?'#b68fdf':'url(#pcTileTexture)')+'" stroke="'+(select?'#ffcb6b':'#d9e5f2')+'" stroke-width="'+(select?'3.4':'1.5')+'"/>'+
    (actions.has(i)?'<circle cx="'+(x+18)+'" cy="'+(y-18)+'" r="4.5" fill="#e879b5" stroke="white" stroke-width="1"/>':'')+
    (decos.has(i)?'<circle cx="'+(x-18)+'" cy="'+(y-18)+'" r="3.5" fill="#87d4b2"/>':'')+
    '<circle cx="'+x+'" cy="'+y+'" r="24" fill="transparent" stroke="none" pointer-events="all"/></g>';
 }
 html+='</g>'+balls();
 const viewBox=svg.getAttribute('viewBox');S.writing=true;svg.innerHTML=html;if(viewBox)svg.setAttribute('viewBox',viewBox);S.writing=false;
 if(window.ADOFAIOriginalTileTexture)svg.querySelector('#pcTileSprite').setAttribute('href',window.ADOFAIOriginalTileTexture);
 S.planets={fire:svg.querySelector('#pcFire'),ice:svg.querySelector('#pcIce'),line:svg.querySelector('#pcOrbitLink')};
 for(const g of svg.querySelectorAll('.chart-floor')){
  const idx=Number(g.getAttribute('data-floor'));
  g.addEventListener('click',()=>api.selectFloor(idx));
  g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();api.selectFloor(idx);}});
 }
 updateBalls();
}
function updateBalls(){
 const s=S.api?.state,p=S.planets;if(!p||!s?.doc)return;
 const player=window.ADOFAIArchivePlayer,playing=!!player?.state?.playing;
 const idx=Math.max(0,Math.min(s.positions.length-1,playing?player.state.playerIndex:s.selected));
 const center=s.positions[idx],previous=s.positions[idx-1],next=s.positions[idx+1];
 if(!center)return;
 const cx=center.x*UNIT,cy=center.y*UNIT;
 // Actual ADOFAI rule: one planet remains centered on the current floor.
 // The other circles it at the distance between two adjacent floors,
 // then the roles switch on landing.
 const twoPi=Math.PI*2;
 let angleToPrevious=previous?Math.atan2((previous.y-center.y),(previous.x-center.x)):NaN;
 let angleToNext=next?Math.atan2((next.y-center.y),(next.x-center.x)):NaN;
 if(!Number.isFinite(angleToPrevious)&&Number.isFinite(angleToNext))angleToPrevious=angleToNext+Math.PI;
 if(!Number.isFinite(angleToNext)&&Number.isFinite(angleToPrevious))angleToNext=angleToPrevious+Math.PI;
 if(!Number.isFinite(angleToPrevious))angleToPrevious=Math.PI;
 if(!Number.isFinite(angleToNext))angleToNext=0;
 let phase=0;
 if(playing){
   const times=player.state.timings,seconds=el('chartAudio')?.currentTime||0;
   const t0=times?.[idx]??seconds,t1=times?.[idx+1]??(t0+.45);
   phase=Math.max(0,Math.min(1,(seconds-t0)/Math.max(.015,t1-t0)));
 }
 let flipped=false;
 for(const floor of (S.twirlFloors||[])){if(floor>idx)break;flipped=!flipped;}
 const travel=flipped?
   ((angleToNext-angleToPrevious+twoPi*3)%twoPi):
   ((angleToPrevious-angleToNext+twoPi*3)%twoPi);
 const turn=travel<0.001?Math.PI:travel;
 const rotation=angleToPrevious+(flipped?1:-1)*turn*phase;
 const mx=cx+Math.cos(rotation)*UNIT,my=cy+Math.sin(rotation)*UNIT;
 const fireFixed=idx%2===0;
 const fx=fireFixed?cx:mx,fy=fireFixed?cy:my;
 const ix=fireFixed?mx:cx,iy=fireFixed?my:cy;
 p.fire.setAttribute('transform','translate('+fx.toFixed(2)+' '+fy.toFixed(2)+')');
 p.ice.setAttribute('transform','translate('+ix.toFixed(2)+' '+iy.toFixed(2)+')');
 for(const [key,value] of Object.entries({x1:fx,y1:fy,x2:ix,y2:iy}))
   p.line.setAttribute(key,value.toFixed(2));
}
function loop(){updateBalls();requestAnimationFrame(loop);}
function boot(){
 S.api=window.ADOFAIChartPort;S.svg=el('chartSvg');if(!S.api||!S.svg)return;
 const css=document.createElement('style');css.textContent='#chartSvg{background:radial-gradient(circle at 50% 40%,#203042,#090f1a)!important}.chart-floor .pc-tile{filter:drop-shadow(0 2px 1px #0008)}.chart-floor:hover .pc-tile{stroke:#fff7d5;stroke-width:3.5}.chart-floor.live .pc-tile{stroke:#ffe6a8;stroke-width:4;filter:drop-shadow(0 0 8px #ffc56a)}.chart-original{display:none!important}';document.head.appendChild(css);
 const obs=new MutationObserver(()=>{if(!S.writing&&S.svg.querySelector('.chart-floor > rect'))draw();});obs.observe(S.svg,{childList:true,subtree:true});
 if(S.api.state.doc)draw();requestAnimationFrame(loop);
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();