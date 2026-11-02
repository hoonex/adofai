/* Tile mesh based on FloorMesh.GetPositions + GetPositions.GeneratePolygon
 * decompiled from the user's PC Unity Assembly-CSharp.dll.
 * Original methods: FloorMesh.GetPositions(float,float,float,float,int),
 * and scrFloor.UpdateAngle which sets entry/exit radians in FloorMesh.
 * Browser path renderer is a reconstruction, NOT the actual Unity runtime.
 * Textured fill from actual user's PC tile_white sprite.
 */
(function(){
'use strict';
const ns='http://www.w3.org/2000/svg',U=54,PI=Math.PI,TAU=2*PI;
const el=id=>document.getElementById(id);
const S={api:null,svg:null,planets:null,writing:false,twirlFloors:[]};
const norm=a=>((a%TAU)+TAU)%TAU;
const modDelta=(a,b)=>norm(b-a);
const V=(x,y)=>({x,y});
const add=(p,q)=>V(p.x+q.x,p.y+q.y);
const sub=(p,q)=>V(p.x-q.x,p.y-q.y);
const scale=(p,t)=>V(p.x*t,p.y*t);
const fromAngle=(a,l)=>V(l*Math.cos(a),l*Math.sin(a));
const lerp=(a,b,t)=>add(scale(a,1-t),scale(b,t));
function lineIntersection(a,b,c,d){
 const r=sub(b,a),s=sub(d,c),cross=r.x*s.y-r.y*s.x;
 if(Math.abs(cross)<1e-7) return V(0,0);
 const diff=sub(c,a),t=(diff.x*s.y-diff.y*s.x)/cross;
 const v=add(a,scale(r,t));
 if(!Number.isFinite(v.x)||!Number.isFinite(v.y))return V(0,0);
 return v;
}
function arc(center,start,end,r,count){
 while(end<start)end+=TAU;
 if(end-start>TAU)end=start+TAU;
 const pts=[];
 for(let i=0;i<count;i++)pts.push(add(center,fromAngle(start+(end-start)*i/Math.max(1,count-1),r)));
 return pts;
}
function curvatureFactor(delta){
 const d=delta*180/PI;
 if(d<5)return 1;
 if(d<30)return 1-(1-.83)*Math.pow((d-5)/25,.5);
 if(d<45)return .83-(.83-.77)*(d-30)/15;
 if(d<90)return .77-(.77-.15)*Math.pow((d-45)/45,.7);
 if(d<120)return .15*Math.pow(1-(d-90)/30,.5);
 return 0;
}
function meshPolygon(prev,curr,next,width=U*0.37,length=U/2){
 // Reconstruct the *outer polygon* based on PC FloorMesh.GetPositions.
 // Separate the 0.5-tile longitudinal join from the slimmer 0.37-tile width,
 // preserving end-cap connections while reducing the visually oversized track.
 // Work in y-up Unity coordinates, reflecting the browser's y-down positions.
 const origin=V(0,0);
 function direction(p,fallback){
  if(!p||Math.hypot(p.x-curr.x,p.y-curr.y)<.00001)return fallback;
  return Math.atan2(curr.y-p.y,p.x-curr.x);
 }
 let a=direction(prev,NaN),b=direction(next,NaN);
 if(!Number.isFinite(a)&&!Number.isFinite(b)){a=PI;b=0;}
 else if(!Number.isFinite(a))a=b+PI;
 else if(!Number.isFinite(b))b=a+PI;
 a=norm(a);b=norm(b);
 if(modDelta(a,b)>=PI){const t=a;a=b;b=t;}
 const short=Math.min(modDelta(a,b),modDelta(b,a));
 const zero=short<.0001,straight=Math.abs(short-PI)<.0001;
 let segment=length,base=origin;
 if(zero){base=add(base,fromAngle(b,-segment/12));segment*=2/3;}
 const p0=add(base,fromAngle(a,segment));
 const p1=add(base,fromAngle(b,segment));
 const sm=add(p0,fromAngle(a+PI/2,width));
 const sl=add(p0,fromAngle(a-PI/2,width));
 const em=add(p1,fromAngle(b+PI/2,width));
 const ee=add(p1,fromAngle(b-PI/2,width));
 const near=.01*width;
 const sm2=add(sm,fromAngle(a+PI,near));
 const sl2=add(sl,fromAngle(a-PI,near));
 const em2=add(em,fromAngle(b+PI,near));
 const ee2=add(ee,fromAngle(b-PI,near));
 const ccw=(zero||straight)?base:lineIntersection(sm,sm2,ee,ee2);
 const cw=(zero||straight)?base:lineIntersection(sl,sl2,em,em2);
 const factor=curvatureFactor(short);
 const delta=modDelta(a,b);
 const rightCenter=lerp(cw,base,factor);
 const leftCenter=lerp(ccw,base,factor);
 const radius=width*factor;
 const pt0=delta<2*PI/3 ?
    arc(rightCenter,norm(b+PI/2),norm(a-PI/2),radius,Math.abs(radius)>1?13:2):
    [cw];
 const pt1=delta>250*PI/180 ?
    arc(leftCenter,norm(a+PI/2),norm(b-PI/2),radius,Math.abs(radius)>1?13:2):
    [ccw];
 const activeCross=delta<PI?ccw:cw;
 const overlap=Math.hypot(sm.x,sm.y)<Math.hypot(activeCross.x,activeCross.y);
 let outline;
 if(zero){outline=[sm,...pt0,ee];}
 else if(straight){outline=[sm,ee,em,sl];}
 else if(!overlap){outline=[sm,...pt1,ee,em,...pt0,sl];}
 else {
  const t0=lineIntersection(sl,sm,ee,ee2);
  const t1=lineIntersection(sl,sm,ee,em);
  const t2=lineIntersection(ee,em,sm,sm2);
  outline=[sm,t2,em,...pt0,sl,t0,ee,t1];
 }
 // Guard against accidental degeneracy from double/midspin floors.
 const points=outline.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&Math.abs(p.x)<2*U&&Math.abs(p.y)<2*U);
 if(points.length<3){
  const h=width;return 'M '+(curr.x*U-h)+' '+(curr.y*U-h)+' h '+(2*h)+' v '+(2*h)+' h '+(-2*h)+' Z';
 }
 return 'M '+points.map(p=>(curr.x*U+p.x).toFixed(2)+' '+(curr.y*U-p.y).toFixed(2)).join(' L ')+' Z';
}
function parseColor(v,defaultColor){
 if(typeof v!=='string')return defaultColor;
 const c=v.trim().replace('#','');return /^[0-9a-f]{6}$/i.test(c)?'#'+c:defaultColor;
}
const defs='<defs><radialGradient id="adoFire" cx="28%" cy="25%" r="88%"><stop stop-color="#fff5ec"/><stop offset=".32" stop-color="#ffac6f"/><stop offset=".75" stop-color="#f24b38"/><stop offset="1" stop-color="#a01519"/></radialGradient><radialGradient id="adoIce" cx="28%" cy="25%" r="88%"><stop stop-color="#fff"/><stop offset=".32" stop-color="#aaeeff"/><stop offset=".75" stop-color="#4aa9ec"/><stop offset="1" stop-color="#1765ac"/></radialGradient><pattern id="adoTileWhite" width="54" height="54" patternUnits="userSpaceOnUse"><image width="54" height="54" preserveAspectRatio="none" id="adoTexture"/></pattern><filter id="adoGlow"><feGaussianBlur stdDeviation="4"/></filter></defs>';
function balls(){
 const ball=(id,fill,glow)=>'<g id="'+id+'"><circle r="23" fill="'+glow+'" opacity=".3" filter="url(#adoGlow)"/><circle r="17" fill="url(#'+fill+')" stroke="#f0f7ff" stroke-width="2.2"/><circle cx="-5" cy="-7" r="4.5" fill="#fff" opacity=".68"/></g>';
 return '<g id="adoPlanets" style="pointer-events:none"><line id="adoLink" stroke="#f8fcff" stroke-opacity=".37" stroke-width="2"/>'+ball('pcFire','adoFire','#ff4040')+ball('pcIce','adoIce','#44bfff')+'</g>';
}
function draw(){
 const api=S.api,svg=S.svg,s=api?.state;
 if(!s?.doc||!svg)return;
 const ps=s.positions||[];if(!ps.length)return;
 const selected=s.selected||0, actions=new Set((s.doc.actions||[]).filter(e=>Number.isInteger(e?.floor)).map(e=>e.floor));
 const decos=new Set((s.doc.decorations||[]).filter(e=>Number.isInteger(e?.floor)).map(e=>e.floor));
 S.twirlFloors=(s.doc.actions||[]).filter(e=>e?.eventType==='Twirl'&&Number.isInteger(e.floor)).map(e=>e.floor).sort((a,b)=>a-b);
 S.radiusEvents=(s.doc.actions||[]).filter(e=>e?.eventType==='ScaleRadius'&&Number.isInteger(e.floor)&&Number.isFinite(Number(e.scale))).sort((a,b)=>a.floor-b.floor);
 S.planetEvents=(s.doc.actions||[]).filter(e=>e?.eventType==='ScalePlanets'&&Number.isInteger(e.floor)&&Number.isFinite(Number(e.scale))).sort((a,b)=>a.floor-b.floor);
 const colorEvents=(s.doc.actions||[]).filter(e=>e?.eventType==='ColorTrack'&&Number.isInteger(e.floor)).sort((a,b)=>a.floor-b.floor);
 let colorIndex=0,activeColor=s.doc.settings?.trackColor||'ffffff',activeType=s.doc.settings?.trackColorType||'Single';
 const start=ps.length>6500?Math.max(0,selected-2300):0,end=ps.length>6500?Math.min(ps.length,selected+2300):ps.length;
 // Beveled outlines are computed from actual entry/exit floor angles,
 // not identical rounded squares and not a thick polyline underneath.
 let html=defs+'<g id="adoFloorMeshes">';
 const color=parseColor(s.doc.settings?.trackColor,'#e4e8f4');
 const border=parseColor(s.doc.settings?.secondaryTrackColor,'#95a1b6');
 for(let i=start;i<end;i++){
  const p=ps[i];if(!p)continue;
  const path=meshPolygon(ps[i-1],p,ps[i+1]);
  const active=i===selected,mid=i>0&&s.angles[i-1]===999;
  let oneShot=null;
  while(colorIndex<colorEvents.length&&colorEvents[colorIndex].floor<=i){
    const e=colorEvents[colorIndex++];
    if(e.justThisTile){if(e.floor===i)oneShot=e;}
    else {activeColor=e.trackColor||activeColor;activeType=e.trackColorType||activeType;}
  }
  const c=oneShot?.trackColor||activeColor,mode=oneShot?.trackColorType||activeType;
  const fill=mid?'#b69ed6':mode==='Rainbow'?'hsl('+((i*16)%360)+' 67% 65%)':parseColor(c,'#d0dbea');
  html+='<g class="chart-floor'+(active?' selected-floor':'')+'" data-floor="'+i+'" tabindex="0" role="button" aria-label="타일 '+i+'" style="cursor:pointer">'+
   '<path class="pc-tile" d="'+path+'" fill="'+fill+'" stroke="'+(active?'#f9d48c':border)+'" stroke-width="'+(active?2:1.1)+'" stroke-linejoin="round"/>'+
   (mid?'':'<path d="'+path+'" fill="url(#adoTileWhite)" opacity=".29" stroke="none" pointer-events="none"/>')+
   (actions.has(i)?'<circle cx="'+(p.x*U+9)+'" cy="'+(p.y*U-9)+'" r="3" fill="#f38db6" stroke="#fff" stroke-width=".8"/>':'')+
   (decos.has(i)?'<circle cx="'+(p.x*U-10)+'" cy="'+(p.y*U-10)+'" r="3" fill="#7cd2a6"/>':'')+
   '<circle cx="'+(p.x*U)+'" cy="'+(p.y*U)+'" r="19" fill="transparent" pointer-events="all"/></g>';
 }
 html+='</g>'+balls();
 S.writing=true;
 const viewBox=svg.getAttribute('viewBox');
 svg.innerHTML=html;
 if(viewBox)svg.setAttribute('viewBox',viewBox);
 S.writing=false;
 const img=svg.querySelector('#adoTexture');if(img&&window.ADOFAIOriginalTileTexture)img.setAttribute('href',window.ADOFAIOriginalTileTexture);
 S.planets={fire:svg.querySelector('#pcFire'),ice:svg.querySelector('#pcIce'),line:svg.querySelector('#adoLink')};
 for(const g of svg.querySelectorAll('.chart-floor')){
  const index=Number(g.getAttribute('data-floor'));
  g.addEventListener('click',()=>api.selectFloor(index));
  g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();api.selectFloor(index);}});
 }
 updateBalls();
}
function updateBalls(){
 const state=S.api?.state,parts=S.planets;if(!state?.doc||!parts)return;
 const pl=window.ADOFAIArchivePlayer;
 const playing=!!pl?.state?.playing;
 const idx=Math.max(0,Math.min(state.positions.length-1,playing?pl.state.playerIndex:state.selected));
 const pos=state.positions[idx];if(!pos)return;
 const x=pos.x*U,y=pos.y*U;
 const timing=pl?.state;
 // Use the SAME travel and Twirl direction as the timeline. Previously, on
 // a 90-degree turn the SVG could rotate 270 degrees during a 90-degree beat,
 // falsely making an unchanged BPM appear 3x faster.
 let incomingDeg=timing?.entryAngleDegrees?.[idx];
 if(!Number.isFinite(incomingDeg)){
  const previous=state.angles?.[idx-1]??state.angles?.[0]??0;
  incomingDeg=180-previous;
 }
 const travelDeg=timing?.travelDegrees?.[idx]??180;
 const reverse=!!timing?.twirlAt?.[idx];
 let phase=0;
 if(playing){
  const seconds=el('chartAudio')?.currentTime||0;
  const start=timing?.rotationStart?.[idx]??timing?.timings?.[idx]??seconds;
  const finish=timing?.timings?.[idx+1]??start;
  // Pause and Hold events: stationary until the actual rotation begins.
  if(finish>start+0.00001)phase=Math.max(0,Math.min(1,(seconds-start)/(finish-start)));
 }
 const angle=(incomingDeg+(reverse?-travelDeg:travelDeg)*phase)*PI/180;
 let radius=100;
 for(const e of S.radiusEvents||[]){if(e.floor>idx)break;radius=Number(e.scale);}
 const distance=U*Math.max(0,Math.min(5,radius/100));
 const ox=x+Math.cos(angle)*distance,oy=y+Math.sin(angle)*distance;
 let scalePlanets=100;
 for(const e of S.planetEvents||[]){if(e.floor>idx)break;scalePlanets=Number(e.scale);}
 const visualScale=Math.max(0,Math.min(3,scalePlanets/100));
 parts.fire.setAttribute('data-scale',String(visualScale));
 parts.ice.setAttribute('data-scale',String(visualScale));
 const fixedFire=idx%2===0;
 const fx=fixedFire?x:ox,fy=fixedFire?y:oy,ix=fixedFire?ox:x,iy=fixedFire?oy:y;
 parts.fire.setAttribute('transform','translate('+fx.toFixed(2)+' '+fy.toFixed(2)+') scale('+visualScale.toFixed(3)+')');
 parts.ice.setAttribute('transform','translate('+ix.toFixed(2)+' '+iy.toFixed(2)+') scale('+visualScale.toFixed(3)+')');
 for(const [key,value] of Object.entries({x1:fx,y1:fy,x2:ix,y2:iy}))parts.line.setAttribute(key,value.toFixed(2));
}
function loop(){updateBalls();requestAnimationFrame(loop);}
function boot(){
 S.api=window.ADOFAIChartPort;S.svg=el('chartSvg');if(!S.api||!S.svg)return;
 const css=document.createElement('style');
 css.textContent='#chartSvg{background:radial-gradient(ellipse at 48% 45%,#222b37,#090d16)!important}.chart-floor .pc-tile{filter:drop-shadow(0 1px .5px #0005)}.chart-floor:hover .pc-tile{stroke:#fff4b6;stroke-width:2.4}.chart-floor.live .pc-tile{stroke:#fcdc7b;stroke-width:3;filter:drop-shadow(0 0 5px #ffc65b)}.chart-original{display:none!important}';
 document.head.appendChild(css);
 new MutationObserver(()=>{if(!S.writing&&S.svg.querySelector('.chart-floor > rect'))draw();}).observe(S.svg,{childList:true,subtree:true});
 if(S.api.state.doc)draw();
 requestAnimationFrame(loop);
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();