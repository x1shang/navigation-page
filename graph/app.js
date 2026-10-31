/* Function explorer. No external runtime or network dependency. */
(() => {
  'use strict';
  const Lab = window.FunctionLab, $ = id => document.getElementById(id);
  const colors = {blue:'#75aaff',green:'#60d5ae',pink:'#f18ab6',yellow:'#ffce73'};
  const state = {group:'trig',id:'sec',mode:'original',x:.65,C:0,a:0,b:1,view:[],playing:false,frame:0};
  let chartKey = '', geomFrame = 0;
  const fn = () => Lab.byId[state.id];
  const fmt = (n, digits=4) => !Number.isFinite(n) ? '无有限实数值' : (Math.abs(n)<.5*10**-digits ? '0' : Number(n.toFixed(digits)).toString());
  const nattr = n => Number(n.toFixed(3));
  const pathPoint = (x,y) => `${nattr(x)},${nattr(y)}`;
  const text = (x,y,t,extra='') => `<text x="${nattr(x)}" y="${nattr(y)}" ${extra}>${t}</text>`;
  const line = (x1,y1,x2,y2,extra='') => `<line x1="${nattr(x1)}" y1="${nattr(y1)}" x2="${nattr(x2)}" y2="${nattr(y2)}" ${extra}/>`;
  const dot = (x,y,color,r=5) => `<circle cx="${nattr(x)}" cy="${nattr(y)}" r="${r}" fill="${color}" class="point"/>`;
  const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
  let layout;
  function getLayout() {
    const W = Math.max(500,Math.round($('plot').clientWidth)), H = W<650 ? 350 : 460;
    const box = {left:W<650?52:58,top:23,right:W-22,bottom:H-42};
    const [xmin,xmax,ymin,ymax] = state.view;
    return {W,H,box,xmin,xmax,ymin,ymax,
      X:x=>box.left+(x-xmin)/(xmax-xmin)*(box.right-box.left),
      Y:y=>box.bottom-(y-ymin)/(ymax-ymin)*(box.bottom-box.top)};
  }
  function niceStep(span, count=7) {
    const raw=span/count, power=10**Math.floor(Math.log10(raw));
    return [1,2,2.5,5,10].find(v=>v*power>=raw)*power;
  }
  function piLabel(x) {
    const n = Math.round(2*x/Math.PI);
    if(n===0)return '0';
    const sign=n<0?'−':'',a=Math.abs(n);
    return sign+(a===2?'π':a%2===0?`${a/2}π`:a===1?'π/2':`${a}π/2`);
  }
  function axes() {
    const {W,H,box:b,X,Y,xmin,xmax,ymin,ymax}=layout;
    const trig=(fn().group==='trig'||fn().id==='sec-square') && xmax-xmin>=5 && xmax-xmin<=28;
    const sx=trig?(xmax-xmin>15?Math.PI:Math.PI/2):niceStep(xmax-xmin),sy=niceStep(ymax-ymin,6);
    let out=`<defs><clipPath id="plot-clip"><rect x="${b.left}" y="${b.top}" width="${b.right-b.left}" height="${b.bottom-b.top}"/></clipPath></defs>`;
    const font=W<650?'style="font-size:14px"':'';
    for(let x=Math.ceil(xmin/sx)*sx;x<=xmax+sx*1e-6;x+=sx){
      out+=line(X(x),b.top,X(x),b.bottom,'class="grid-line"');
      out+=text(X(x),b.bottom+23,trig?piLabel(x):fmt(x,3),`class="tick" text-anchor="middle" ${font}`);
    }
    for(let y=Math.ceil(ymin/sy)*sy;y<=ymax+sy*1e-6;y+=sy){
      out+=line(b.left,Y(y),b.right,Y(y),'class="grid-line"');
      out+=text(b.left-10,Y(y)+4,fmt(y,3),`class="tick" text-anchor="end" ${font}`);
    }
    if(xmin<=0&&xmax>=0)out+=line(X(0),b.top,X(0),b.bottom,'class="axis"');
    if(ymin<=0&&ymax>=0)out+=line(b.left,Y(0),b.right,Y(0),'class="axis"');
    out+=text(b.right-3,b.bottom-8,'x',`class="tick" text-anchor="end" ${font}`);
    out+=text(b.left+10,b.top+10,'y',`class="tick" ${font}`);
    return out;
  }
  function sampleCurve(mode, reflected=false) {
    const f=fn(),{xmin,xmax,ymin,ymax,X,Y}=layout;
    let min=reflected?Math.min(xmin,ymin):xmin,max=reflected?Math.max(xmax,ymax):xmax;
    if(reflected&&f.inverseLimit){min=Math.max(min,f.inverseLimit[0]);max=Math.min(max,f.inverseLimit[1]);}
    if(min>max)return '';
    const samples=Array.from({length:1501},(_,i)=>min+(max-min)*i/1500);
    for(const edge of f.edges)if(edge>=min&&edge<=max)samples.push(edge);
    samples.sort((a,b)=>a-b);
    let path='',lastBranch=null,connected=false;
    for(const x of samples){
      const branch=f.branch(x),y=Lab.value(f,mode,x,state.C);
      if(branch===null||!Number.isFinite(y)){connected=false;lastBranch=null;continue;}
      const px=reflected?X(y):X(x),py=reflected?Y(x):Y(y);
      // Avoid enormous SVG coordinates close to a pole; never join across branches.
      if(Math.abs(px)>1e6||Math.abs(py)>1e6){connected=false;continue;}
      path+=(connected&&branch===lastBranch?'L':'M')+pathPoint(px,py);
      lastBranch=branch;connected=true;
    }
    return path;
  }
  function asymptotes() {
    const f=fn(),{xmin,xmax,X,Y,box:b}=layout;
    let poles=[];
    if(['tan','sec','sec-square'].includes(f.id)) {
      for(let k=Math.ceil((xmin-Math.PI/2)/Math.PI);k<=(xmax-Math.PI/2)/Math.PI;k++)poles.push(Math.PI/2+k*Math.PI);
    }else if(['cot','csc'].includes(f.id)){
      for(let k=Math.ceil(xmin/Math.PI);k<=xmax/Math.PI;k++)poles.push(k*Math.PI);
    }else if(['csch','coth','acsch','asech'].includes(f.id))poles=[0];
    else if(['atanh','acoth','root-circle','root-minus','rational-minus'].includes(f.id))poles=[-1,1];
    return poles.filter(x=>x>=xmin&&x<=xmax).map(x=>line(X(x),b.top,X(x),b.bottom,'class="asymptote"')).join('')+
      (f.horizontal||[]).map(y=>line(b.left,Y(y),b.right,Y(y),'class="asymptote"')).join('');
  }
  function ensureChart() {
    layout=getLayout();
    const key=JSON.stringify([state.id,state.mode,state.view,state.C,$('inverse').checked,layout.W]);
    if(key===chartKey)return;
    chartKey=key;
    const f=fn(),{W,H,box:b,X,Y,xmin,xmax,ymin,ymax}=layout;
    $('plot').setAttribute('viewBox',`0 0 ${W} ${H}`);
    $('plot').setAttribute('aria-label',`${f.label} 的${state.mode==='derivative'?'导函数与切线':state.mode==='integral'?'原函数与有符号积分':'函数图像'}，用左右方向键移动观察点`);
    let content=axes()+`<g clip-path="url(#plot-clip)">${asymptotes()}<g id="area-layer"></g>`;
    if(state.mode==='original'&&$('inverse').checked&&f.inverse){
      content+=line(X(Math.min(xmin,ymin)),Y(Math.min(xmin,ymin)),X(Math.max(xmax,ymax)),Y(Math.max(xmax,ymax)),'class="guide" style="stroke:var(--yellow)"');
      content+=`<path class="curve reflected" d="${sampleCurve('original',true)}"/>`;
      if(f.inverseLimit){
        const [lo,hi]=f.inverseLimit;
        const left=clamp(X(lo),b.left,b.right),right=clamp(X(hi),b.left,b.right);
        if(right>left)content+=`<rect x="${left}" y="${b.top}" width="${right-left}" height="${b.bottom-b.top}" fill="#ffce7306"/>`;
      }
    }
    content+=`<path class="curve" d="${sampleCurve('original')}"/>`;
    if(state.mode!=='original')content+=`<path class="curve ${state.mode}" d="${sampleCurve(state.mode)}"/>`;
    content+='<g id="dynamic-layer"></g></g>';
    $('plot').innerHTML=`<title>${f.label} · 交互函数图像</title>`+content;
    $('scrub').min=state.view[0];$('scrub').max=state.view[1];
    $('view-status').textContent=`x ∈ [${fmt(xmin,2)}, ${fmt(xmax,2)}]`;
  }
  function area() {
    if(state.mode!=='integral')return '';
    const f=fn(),result=Lab.integral(f,state.a,state.b);
    if(!result.ok)return '';
    const {X,Y}=layout,lo=Math.min(state.a,state.b),hi=Math.max(state.a,state.b),direction=Math.sign(state.b-state.a);
    // Fill only the visible part; numerical values still use the full interval.
    const start=Math.max(lo,layout.xmin),end=Math.min(hi,layout.xmax);
    if(end<start)return '';
    let positive='',negative='';
    function piece(x0,y0,x1,y1){
      const p=`M${pathPoint(X(x0),Y(0))}L${pathPoint(X(x0),Y(y0))}L${pathPoint(X(x1),Y(y1))}L${pathPoint(X(x1),Y(0))}Z`;
      if((y0+y1)*direction>=0)positive+=p;else negative+=p;
    }
    const n=360;
    for(let i=0;i<n;i++){
      const x0=start+(end-start)*i/n,x1=start+(end-start)*(i+1)/n,y0=f.f(x0),y1=f.f(x1);
      if(!Number.isFinite(y0)||!Number.isFinite(y1))continue;
      if(y0*y1<0){const root=x0+(x1-x0)*(-y0)/(y1-y0);piece(x0,y0,root,0);piece(root,0,x1,y1);}else piece(x0,y0,x1,y1);
    }
    return `<path d="${positive}" fill="${colors.green}" fill-opacity=".17"/><path d="${negative}" fill="${colors.pink}" fill-opacity=".17"/>`;
  }
  function updateValues() {
    const f=fn(),x=state.x,original=Lab.value(f,'original',x),derived=Lab.value(f,'derivative',x),current=Lab.value(f,state.mode,x,state.C);
    $('value').textContent=fmt(current);
    $('secondary').textContent=fmt(state.mode==='original'?derived:original);
    $('value').style.color=state.mode==='derivative'?colors.green:state.mode==='integral'?colors.pink:colors.blue;
    if(document.activeElement!==$('point-x'))$('point-x').value=fmt(x,5);
    $('scrub').value=x;
    let status=`x = ${fmt(x)} · ${state.mode==='derivative'?'f′(x)':state.mode==='integral'?'F(x) + C':'f(x)'} = ${fmt(current)}`;
    if(!Number.isFinite(current))status=Number.isFinite(original)&&state.mode==='derivative'?'此点的导数不是有限实数；原函数在此有值。':'此点无有限实数值，请移动到定义域内。';
    else if(current<layout.ymin||current>layout.ymax)status+=' · 观察点超出纵轴视窗，可缩小坐标轴';
    else if(x<layout.xmin||x>layout.xmax)status+=' · 观察点超出横轴视窗';
    $('point-status').textContent=status;
    const {X,Y,box:b}=layout;
    let out='';
    if(Number.isFinite(x))out+=line(X(x),b.top,X(x),b.bottom,'class="guide"');
    if(Number.isFinite(original)){
      if(state.mode!=='integral'&&Number.isFinite(derived)){
        const span=(layout.xmax-layout.xmin)*.12;
        out+=line(X(x-span),Y(original-derived*span),X(x+span),Y(original+derived*span),'class="tangent"');
      }
      out+=dot(X(x),Y(original),colors.blue,5.5);
    }
    if(state.mode!=='original'&&Number.isFinite(current))out+=dot(X(x),Y(current),state.mode==='derivative'?colors.green:colors.pink,6);
    if(state.mode==='integral'){
      if(Number.isFinite(state.a))out+=line(X(state.a),b.top,X(state.a),b.bottom,'class="guide" style="stroke:var(--pink)"');
      if(document.activeElement!==$('upper'))$('upper').value=Number.isFinite(state.b)?fmt(state.b,5):'';
      const result=Lab.integral(f,state.a,state.b);
      $('integral-result').classList.toggle('error',!result.ok);
      $('integral-result').textContent=result.ok?`∫[${fmt(state.a,3)}, ${fmt(state.b,3)}] f(x) dx = ${fmt(result.value,6)}`:result.message;
    }
    $('dynamic-layer').innerHTML=out;
    $('area-layer').innerHTML=area();
  }
  function render(){ensureChart();updateValues();}
  function buildFamilies() {
    $('families').innerHTML=Lab.groups.map(g=>`<button data-group="${g.id}" aria-pressed="${g.id===state.group}">${g.name}</button>`).join('');
    $('functions').innerHTML=Lab.functions.filter(f=>f.group===state.group).map(f=>`<button data-function="${f.id}" aria-pressed="${f.id===state.id}">${f.label}</button>`).join('');
  }
  function syncInfo() {
    const f=fn(),mode=state.mode;
    $('family-name').textContent=Lab.groups.find(g=>g.id===f.group).en;
    $('function-title').textContent=f.label;
    $('definition').textContent=f.definition===f.label?'y = '+f.label:f.label+' = '+f.definition;
    $('mode-name').textContent=mode==='original'?'观察函数':mode==='derivative'?'斜率 → 导函数':'面积 → 原函数';
    $('formula-label').textContent=mode==='original'?'函数表达式':mode==='derivative'?'点击求导 · 解析结果':'点击积分 · 一组原函数';
    $('formula').textContent=mode==='original'?'f(x) = '+f.label:mode==='derivative'?'f′(x) = '+f.derivative:'∫ f(x) dx = '+f.primitive+' + C';
    $('formula').style.color=mode==='original'?colors.blue:mode==='derivative'?colors.green:colors.pink;
    $('operation-help').textContent=mode==='original'?'蓝点沿函数移动，黄色虚线是该点的切线。':mode==='derivative'?'绿色曲线的高度，等于蓝色曲线在同一 x 处的切线斜率。':'粉色曲线的导数是蓝色函数；拖动 C 可以上下平移整条原函数曲线。';
    $('value-label').textContent=mode==='derivative'?'导函数 f′(x)':mode==='integral'?'原函数 F(x) + C':'函数 f(x)';
    $('secondary-label').textContent=mode==='original'?'切线斜率 f′(x)':'被'+(mode==='integral'?'积':'求导')+'函数 f(x)';
    for(const key of ['domain','range','parity','behavior'])$(key).textContent=f[key];
    $('function-note').textContent=f.note;
    $('integral-controls').hidden=mode!=='integral';
    $('inverse-control').hidden=mode!=='original'||!f.inverse;
    $('operation-legend').hidden=mode==='original';
    $('operation-legend').innerHTML=mode==='derivative'?'<i class="dot green"></i>f′(x) <span class="yellow"> / 虚线：切线</span>':'<i class="dot pink"></i>F(x) + C <span class="green"> / 正贡献</span><span class="pink"> · 负贡献</span>';
    document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mode===mode));
    $('mode-tip').textContent=mode==='integral'?'把积分表变成实验：在「常用被积函数」中切换根式与有理式，查看解析原函数；修改 a、b 后点击「计算 ∫」，或直接拖动观察点改变 b。':mode==='derivative'?'观察黄色切线：斜率为正，绿线在横轴上方；斜率为负，绿线在横轴下方。函数端点有值，也不一定有有限的导数。':$('inverse').checked&&f.inverse?`黄色虚线：${f.inverse}。${f.inverseLimit?'仅将浅色区域中的主值分支':'将当前函数'}关于 y = x 反射。`:'试一试：点击「求导」观察切线斜率，点击「积分」探索原函数和面积；打开「对照反函数」，观察主值分支的对称关系。';
  }
  function stop() {
    state.playing=false;cancelAnimationFrame(state.frame);$('play').textContent='▶ 自动走一遍';$('play').setAttribute('aria-label','播放观察点动画');$('play').setAttribute('aria-pressed','false');
  }
  function choose(id) {
    stop();const f=Lab.byId[id];
    state.id=id;state.group=f.group;state.view=[...f.view];state.C=0;state.a=f.interval[0];state.b=f.interval[1];state.x=state.mode==='integral'?state.b:f.point;
    $('inverse').checked=false;$('lower').value=state.a;$('upper').value=state.b;$('constant').value=0;$('constant-value').textContent='0';
    buildFamilies();syncInfo();render();
  }
  function setMode(mode) {
    stop();state.mode=mode;
    if(mode==='integral'){
      state.a=fn().interval[0];state.b=fn().interval[1];state.x=state.b;$('lower').value=state.a;$('upper').value=state.b;
    }
    syncInfo();render();
  }
  function setX(x) {
    if(!Number.isFinite(x))return;
    state.x=x;if(state.mode==='integral')state.b=x;
    render();
  }
  $('families').addEventListener('click',e=>{const b=e.target.closest('[data-group]');if(b)choose(Lab.functions.find(f=>f.group===b.dataset.group).id);});
  $('functions').addEventListener('click',e=>{const b=e.target.closest('[data-function]');if(b)choose(b.dataset.function);});
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
  $('inverse').addEventListener('change',()=>{syncInfo();render();});
  $('scrub').addEventListener('input',()=>{stop();setX($('scrub').valueAsNumber);});
  $('point-x').addEventListener('input',()=>{stop();setX($('point-x').valueAsNumber);});
  $('point-x').addEventListener('change',()=>{stop();setX($('point-x').valueAsNumber);});
  $('point-x').addEventListener('blur',()=>updateValues());
  $('constant').addEventListener('input',()=>{state.C=+$('constant').value;$('constant-value').textContent=fmt(state.C,1);render();});
  function calculate(){stop();state.a=$('lower').valueAsNumber;state.b=$('upper').valueAsNumber;if(Number.isFinite(state.b))state.x=state.b;render();}
  $('calculate').addEventListener('click',calculate);
  for(const id of ['lower','upper']){
    $(id).addEventListener('focus',stop);
    $(id).addEventListener('keydown',e=>{if(e.key==='Enter')calculate();});
  }
  function zoom(scale){stop();const [x0,x1,y0,y1]=state.view;if((x1-x0)*scale<.15||(x1-x0)*scale>90)return;const cx=(x0+x1)/2,cy=(y0+y1)/2;state.view=[cx+(x0-cx)*scale,cx+(x1-cx)*scale,cy+(y0-cy)*scale,cy+(y1-cy)*scale];render();}
  $('zoom-in').addEventListener('click',()=>zoom(.75));$('zoom-out').addEventListener('click',()=>zoom(1/.75));$('reset').addEventListener('click',()=>choose(state.id));
  let dragging=false;
  function pointerX(e){const rect=$('plot').getBoundingClientRect();return layout.xmin+(clamp((e.clientX-rect.left)/rect.width*layout.W,layout.box.left,layout.box.right)-layout.box.left)/(layout.box.right-layout.box.left)*(layout.xmax-layout.xmin);}
  $('plot').addEventListener('pointerdown',e=>{if(e.button!==0)return;stop();dragging=true;$('plot').setPointerCapture(e.pointerId);setX(pointerX(e));});
  $('plot').addEventListener('pointermove',e=>{if(dragging)setX(pointerX(e));});
  $('plot').addEventListener('pointerup',()=>{dragging=false;});$('plot').addEventListener('pointercancel',()=>{dragging=false;});
  $('plot').addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();stop();setX(state.x+(e.key==='ArrowRight'?1:-1)*(layout.xmax-layout.xmin)/(e.shiftKey?30:200));}if(e.key==='Home'){e.preventDefault();stop();setX(fn().point);}});
  $('play').addEventListener('click',()=>{
    if(state.playing){stop();return;}
    const f=fn(),start=state.view[0],span=state.view[1]-start;
    const points=Array.from({length:1801},(_,i)=>start+span*i/1800).filter(x=>
      Number.isFinite(Lab.value(f,state.mode,x,state.C)) && (state.mode!=='integral'||Lab.integral(f,state.a,x).ok));
    if(!points.length){$('point-status').textContent='当前视窗内没有可播放的有效点，请重置视图或调整积分下限。';return;}
    state.playing=true;$('play').textContent='Ⅱ 暂停';$('play').setAttribute('aria-label','暂停观察点动画');$('play').setAttribute('aria-pressed','true');
    const started=performance.now();
    let last=0;
    function tick(now){
      if(!state.playing)return;
      const p=Math.min((now-started)/10000,1);
      if(now-last>40||p===1){setX(points[Math.round(p*(points.length-1))]);last=now;}
      if(p<1)state.frame=requestAnimationFrame(tick);else stop();
    }
    state.frame=requestAnimationFrame(tick);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  new ResizeObserver(()=>{chartKey='';render();}).observe($('plot').parentElement);

  // Geometry uses one equal scale on both axes. Points and curves share the exact parameterization.
  function geometryAxes(X,Y,width=500,height=330) {
    return line(28,Y(0),width-24,Y(0),'class="axis"')+line(X(0),24,X(0),height-20,'class="axis"')+
      text(width-27,Y(0)-8,'x')+text(X(0)+10,30,'y')+text(X(0)-15,Y(0)+18,'O');
  }
  function drawCircle() {
    const theta=+$('theta').value*Math.PI/180,p=Lab.circlePoint(theta),t=Math.tan(theta),u=1/t;
    const s=Math.min(88,218/Math.max(1,t),290/Math.max(1,u)),X=x=>140+s*x,Y=y=>237-s*y;
    let out=geometryAxes(X,Y)+`<circle cx="${X(0)}" cy="${Y(0)}" r="${s}" class="curve" style="stroke:#425a76;stroke-width:1.7"/>`;
    out+=line(X(1),24,X(1),Y(0),'class="guide"')+line(X(0),Y(1),470,Y(1),'class="guide"');
    // Both tangent intersections lie on the same ray OP.
    const end=Math.max(1/p.x,1/p.y);
    out+=line(X(0),Y(0),X(p.x*end),Y(p.y*end),'stroke="#ffce73" stroke-width="1.5" stroke-dasharray="5 4"');
    out+=line(X(0),Y(0),X(p.x),Y(0),'stroke="#75aaff" stroke-width="3"');
    out+=line(X(p.x),Y(0),X(p.x),Y(p.y),'stroke="#f18ab6" stroke-width="3"');
    out+=line(X(1),Y(0),X(1),Y(t),'stroke="#60d5ae" stroke-width="3"');
    out+=line(X(0),Y(1),X(u),Y(1),'stroke="#ffce73" stroke-width="3"');
    const arcEnd={x:.38*Math.cos(theta),y:.38*Math.sin(theta)};
    out+=`<path d="M${pathPoint(X(.38),Y(0))}A${.38*s},${.38*s} 0 0 0 ${pathPoint(X(arcEnd.x),Y(arcEnd.y))}" fill="none" stroke="#91a2ba"/>`;
    out+=dot(X(p.x),Y(p.y),colors.blue,5)+dot(X(1),Y(t),colors.green,4)+dot(X(u),Y(1),colors.yellow,4);
    out+=text(X(p.x)+10,Y(p.y)+17,'P')+text(X(1)+9,Y(t)-9,'T')+text(X(u)+9,Y(1)-9,'U');
    out+=text(X(.5*p.x),Y(0)+19,'cos θ','text-anchor="middle" style="fill:#75aaff"')+text(X(p.x)-9,Y(.5*p.y),'sin θ','text-anchor="end" style="fill:#f18ab6"');
    out+=text(X(1)+8,Y(t/2)+4,'tan θ','style="fill:#60d5ae"')+text(X(u/2),Y(1)-10,'cot θ','text-anchor="middle" style="fill:#ffce73"');
    out+=text(X(.48),Y(.15),'θ')+text(470,310,`P = (${fmt(p.x,3)}, ${fmt(p.y,3)})`,'text-anchor="end"');
    $('circle-plot').innerHTML='<title>单位圆：P、T、U 由同一角度实时计算</title>'+out;
    $('theta-value').textContent=`${fmt(theta,3)} rad`;
    $('circle-values').innerHTML=[['sin',p.y],['cos',p.x],['tan',t],['cot',u],['sec',1/p.x],['csc',1/p.y]].map(([name,v])=>`<span>${name} = ${fmt(v,3)}</span>`).join('');
  }
  function drawHyperbola() {
    const t=+$('hyper-t').value,p=Lab.hyperbolaPoint(t),s=63,X=x=>145+s*x,Y=y=>165-s*y;
    let out='<defs><clipPath id="hyper-clip"><rect x="26" y="20" width="450" height="290"/></clipPath></defs>'+geometryAxes(X,Y);
    let curve='';
    for(let i=0;i<=700;i++){const q=Lab.hyperbolaPoint(-2.4+4.8*i/700);curve+=(i?'L':'M')+pathPoint(X(q.x),Y(q.y));}
    let sector='M'+pathPoint(X(0),Y(0))+'L'+pathPoint(X(1),Y(0));
    for(let i=1;i<=180;i++){const q=Lab.hyperbolaPoint(t*i/180);sector+='L'+pathPoint(X(q.x),Y(q.y));}
    sector+='Z';
    out+=`<g clip-path="url(#hyper-clip)"><path d="${sector}" fill="#ffce73" fill-opacity=".17"/><path d="${curve}" class="curve"/>`;
    out+=line(X(0),Y(0),X(p.x),Y(p.y),'stroke="#ffce73" stroke-width="1.6" stroke-dasharray="5 4"');
    out+=line(X(0),Y(0),X(p.x),Y(0),'stroke="#60d5ae" stroke-width="3"')+line(X(p.x),Y(0),X(p.x),Y(p.y),'stroke="#f18ab6" stroke-width="3"');
    out+=dot(X(p.x),Y(p.y),colors.blue,6)+'</g>';
    out+=text(X(p.x)+13,Y(p.y)+(t>=0?-11:21),'P(cosh t, sinh t)');
    out+=text(X(p.x/2),Y(0)+(t>=0?22:-13),'cosh t','text-anchor="middle" style="fill:#60d5ae"');
    if(Math.abs(t)>.15)out+=text(X(p.x)+12,Y(p.y/2)+4,'sinh t','style="fill:#f18ab6"');
    out+=text(X(1)-5,Y(0)+21,'1','text-anchor="end"');
    out+=text(475,309,`P = (${fmt(p.x,3)}, ${fmt(p.y,3)})`,'text-anchor="end"');
    $('hyperbola-plot').innerHTML='<title>单位双曲线：参数点 P 始终位于曲线上</title>'+out;
    $('hyper-value').textContent=fmt(t,2);
    $('hyperbola-values').innerHTML=[`cosh t = ${fmt(p.x,4)}`,`sinh t = ${fmt(p.y,4)}`,`x² − y² = ${fmt(p.x*p.x-p.y*p.y,6)}`,`有向面积 = ${fmt(t/2,3)}`].map(v=>`<span>${v}</span>`).join('');
  }
  function geometry(){cancelAnimationFrame(geomFrame);geomFrame=requestAnimationFrame(()=>{drawCircle();drawHyperbola();});}
  $('theta').addEventListener('input',geometry);$('hyper-t').addEventListener('input',geometry);
  choose('sec');drawCircle();drawHyperbola();
})();
