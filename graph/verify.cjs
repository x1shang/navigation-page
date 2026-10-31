'use strict';
const assert=require('node:assert/strict');
const Lab=require('./functions.js');
let comparisons=0;
const close=(actual,expected,message,tolerance=3e-5)=>{assert.ok(Math.abs(actual-expected)<=tolerance*(1+Math.abs(expected)),`${message}: ${actual} vs ${expected}`);comparisons++;};
function simpson(f,a,b,n=2000){let s=f(a)+f(b);for(let i=1;i<n;i++)s+=(i%2?4:2)*f(a+(b-a)*i/n);return s*(b-a)/(3*n);}
for(const f of Lab.functions){
  let count=0;
  for(let i=-120;i<=120;i++){
    const x=i*.047+0.01234,h=1e-6,branch=f.branch(x);
    if(branch===null||f.branch(x-h)!==branch||f.branch(x+h)!==branch||!Number.isFinite(f.d(x))||Math.abs(f.d(x))>1500)continue;
    const derivative=(f.f(x+h)-f.f(x-h))/(2*h),primitiveDerivative=(f.F(x+h)-f.F(x-h))/(2*h);
    close(derivative,f.d(x),`${f.id} f' at ${x}`);
    close(primitiveDerivative,f.f(x),`${f.id} F' at ${x}`);
    count++;
  }
  assert.ok(count>5,`${f.id}: insufficient samples`);
  const [a,b]=f.interval,result=Lab.integral(f,a,b);
  assert.ok(result.ok,`${f.id}: valid default interval`);
  close(result.value,simpson(f.f,a,b),`${f.id}: definite integral`,1e-8);
  close(Lab.integral(f,b,a).value,-result.value,`${f.id}: orientation`,1e-12);
  close(Lab.integral(f,a,a).value,0,`${f.id}: zero interval`,1e-12);
}
for(const [id,a,b] of [['tan',0,Math.PI],['sec',0,Math.PI],['cot',-.5,.5],['csc',-.5,.5],['csch',-1,1],['coth',-1,1],['acsch',-1,1],['atanh',-2,2],['acoth',-2,2],['root-circle',-1,1],['root-minus',-2,2],['rational-minus',-2,2],['asech',0,1]])assert.equal(Lab.integral(Lab.byId[id],a,b).ok,false,`${id}: reject invalid integral`);
for(const [id,x] of [['asec',-.5],['acsc',.5],['acos',1.1],['acosh',0],['atanh',1],['asech',-1],['acoth',0],['acsch',0]])assert.ok(Number.isNaN(Lab.value(Lab.byId[id],'original',x)),`${id}: real domain`);
assert.ok(Number.isFinite(Lab.value(Lab.byId.asin,'original',1)));
assert.ok(Number.isNaN(Lab.value(Lab.byId.asin,'derivative',1)));
assert.equal(Lab.byId.acot.f(0),Math.PI/2);
for(const t of [-1.5,-1,-.01,0,.01,1,1.5]){
  const p=Lab.hyperbolaPoint(t);close(p.x*p.x-p.y*p.y,1,'hyperbola membership',1e-12);
  const points=[{x:0,y:0},{x:1,y:0},...Array.from({length:1000},(_,i)=>Lab.hyperbolaPoint(t*(i+1)/1000))];
  let area=0;for(let i=0;i<points.length;i++){const a=points[i],b=points[(i+1)%points.length];area+=a.x*b.y-b.x*a.y;}
  close(area/2,t/2,'hyperbolic sector area',1e-6);
}
for(let deg=15;deg<=75;deg++){
  const t=deg*Math.PI/180,p=Lab.circlePoint(t);
  close(p.x*p.x+p.y*p.y,1,'circle membership',1e-12);
  close(Math.hypot(1,Math.tan(t)),1/p.x,'OT = sec theta',1e-12);
  close(Math.hypot(1/Math.tan(t),1),1/p.y,'OU = csc theta',1e-12);
}
for(const id of ['root-minus','acsch','asec','acsc','acoth']){
  const f=Lab.byId[id];close(Lab.integral(f,-3,-1.5).value,simpson(f.f,-3,-1.5),`${id}: negative branch`,1e-8);
}
console.log(JSON.stringify({functions:Lab.functions.length,numericComparisons:comparisons,invalidIntervals:13,status:'passed'}));
