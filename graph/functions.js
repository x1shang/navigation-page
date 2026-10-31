(function (root) {
  'use strict';
  const P = Math.PI, abs = Math.abs, sqrt = Math.sqrt, log = Math.log;
  const {sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh} = Math;
  const sec = x => 1 / cos(x), csc = x => 1 / sin(x), cot = x => cos(x) / sin(x);
  const sech = x => 1 / cosh(x), csch = x => 1 / sinh(x), coth = x => 1 / tanh(x);
  const acot = x => P / 2 - atan(x), asec = x => acos(1 / x), acsc = x => asin(1 / x);
  const asech = x => acosh(1 / x), acsch = x => asinh(1 / x), acoth = x => atanh(1 / x);
  const all = () => 0;
  const nonzero = x => abs(x) > 1e-12 ? Math.sign(x) : null;
  const inside = x => abs(x) <= 1 ? 0 : null;
  const outside = x => abs(x) >= 1 ? Math.sign(x) : null;
  const openInside = x => abs(x) < 1 ? 0 : null;
  const openOutside = x => abs(x) > 1 ? Math.sign(x) : null;
  const positive = x => x >= 1 ? 0 : null;
  const sechDomain = x => x > 0 && x <= 1 ? 0 : null;
  const sinBranch = x => abs(sin(x)) > 1e-12 ? Math.floor(x / P) : null;
  const cosBranch = x => abs(cos(x)) > 1e-12 ? Math.floor((x + P / 2) / P) : null;
  const groups = [
    {id:'trig', name:'三角函数', en:'TRIGONOMETRIC'},
    {id:'inverse', name:'反三角函数', en:'INVERSE TRIGONOMETRIC'},
    {id:'hyper', name:'双曲函数', en:'HYPERBOLIC'},
    {id:'ihyper', name:'反双曲函数', en:'INVERSE HYPERBOLIC'},
    {id:'kernels', name:'常用被积函数', en:'INTEGRAL PATTERNS'}
  ];
  const functions = [];
  function add(id, group, label, f, d, F, derivative, primitive, domain, range, parity, behavior, branch, options = {}) {
    functions.push({id, group, label, f, d, F, derivative, primitive, domain, range, parity, behavior, branch,
      definition:label, view:[-4,4,-3.5,3.5], point:.65, anchor:0, interval:[0,1], edges:[], note:'所有角度均使用弧度。', ...options});
  }
  const trigView = [-2*P,2*P,-3.5,3.5];
  add('sin','trig','sin x',sin,cos,x=>-cos(x),'cos x','−cos x','ℝ','[−1, 1]','奇函数','周期 2π；分段单调',all,{view:trigView,inverse:'arcsin x',inverseLimit:[-P/2,P/2]});
  add('cos','trig','cos x',cos,x=>-sin(x),sin,'−sin x','sin x','ℝ','[−1, 1]','偶函数','周期 2π；分段单调',all,{view:trigView,inverse:'arccos x',inverseLimit:[0,P]});
  add('tan','trig','tan x',tan,x=>sec(x)**2,x=>-log(abs(cos(x))),'sec²x','−ln|cos x|','x ≠ π/2 + kπ','ℝ','奇函数','每个连续区间上递增；周期 π',cosBranch,{view:trigView,inverse:'arctan x',inverseLimit:[-P/2,P/2],note:'竖直渐近线两侧属于不同分支，不能跨越奇点计算定积分。'});
  add('cot','trig','cot x',cot,x=>-(csc(x)**2),x=>log(abs(sin(x))),'−csc²x','ln|sin x|','x ≠ kπ','ℝ','奇函数','每个连续区间上递减；周期 π',sinBranch,{view:trigView,anchor:.5,interval:[.5,1.5],inverse:'arccot x',inverseLimit:[0,P]});
  add('sec','trig','sec x',sec,x=>sec(x)*tan(x),x=>log(abs(sec(x)+tan(x))),'sec x · tan x','ln|sec x + tan x|','x ≠ π/2 + kπ','(−∞, −1] ∪ [1, +∞)','偶函数','周期 2π；分段单调',cosBranch,{view:trigView,inverse:'arcsec x',inverseLimit:[0,P],definition:'1 / cos x'});
  add('csc','trig','csc x',csc,x=>-csc(x)*cot(x),x=>log(abs(csc(x)-cot(x))),'−csc x · cot x','ln|csc x − cot x|','x ≠ kπ','(−∞, −1] ∪ [1, +∞)','奇函数','周期 2π；分段单调',sinBranch,{view:trigView,anchor:.5,interval:[.5,1.5],inverse:'arccsc x',inverseLimit:[-P/2,P/2],definition:'1 / sin x'});
  add('asin','inverse','arcsin x',asin,x=>1/sqrt(1-x*x),x=>x*asin(x)+sqrt(1-x*x),'1 / √(1 − x²)','x arcsin x + √(1 − x²)','[−1, 1]','[−π/2, π/2]','奇函数','整个定义域上递增',inside,{view:[-1.8,1.8,-2.3,2.3],edges:[-1,1],interval:[-.5,.5],inverse:'sin x',note:'主值在 [−π/2, π/2]；端点处函数有值，但导数不是有限实数。'});
  add('acos','inverse','arccos x',acos,x=>-1/sqrt(1-x*x),x=>x*acos(x)-sqrt(1-x*x),'−1 / √(1 − x²)','x arccos x − √(1 − x²)','[−1, 1]','[0, π]','非奇非偶','整个定义域上递减',inside,{view:[-1.8,1.8,-3.5,3.5],edges:[-1,1],interval:[-.5,.5],inverse:'cos x',note:'主值在 [0, π]；端点处函数有值，但导数不是有限实数。'});
  add('atan','inverse','arctan x',atan,x=>1/(1+x*x),x=>x*atan(x)-.5*log(1+x*x),'1 / (1 + x²)','x arctan x − ½ ln(1 + x²)','ℝ','(−π/2, π/2)','奇函数','整个定义域上递增',all,{inverse:'tan x',horizontal:[-P/2,P/2]});
  add('acot','inverse','arccot x',acot,x=>-1/(1+x*x),x=>x*acot(x)+.5*log(1+x*x),'−1 / (1 + x²)','x arccot x + ½ ln(1 + x²)','ℝ','(0, π)','非奇非偶','整个定义域上递减',all,{inverse:'cot x',definition:'π/2 − arctan x',horizontal:[P],note:'本页约定 arccot x = π/2 − arctan x，主值为 (0, π)，在 x = 0 处连续。'});
  add('asec','inverse','arcsec x',asec,x=>1/(abs(x)*sqrt(x*x-1)),x=>x*asec(x)-acosh(abs(x)),'1 / (|x| √(x² − 1))','x arcsec x − arcosh|x|','(−∞, −1] ∪ [1, +∞)','[0, π/2) ∪ (π/2, π]','非奇非偶','两个分支上分别递增',outside,{point:1.7,anchor:1.2,interval:[1.2,2],edges:[-1,1],inverse:'sec x',definition:'arccos(1/x)',note:'绝对值 |x| 不能省略；两个连通分支上的积分常数可以各自选取。'});
  add('acsc','inverse','arccsc x',acsc,x=>-1/(abs(x)*sqrt(x*x-1)),x=>x*acsc(x)+acosh(abs(x)),'−1 / (|x| √(x² − 1))','x arccsc x + arcosh|x|','(−∞, −1] ∪ [1, +∞)','[−π/2, 0) ∪ (0, π/2]','奇函数','两个分支上分别递减',outside,{point:1.7,anchor:1.2,interval:[1.2,2],edges:[-1,1],inverse:'csc x',definition:'arcsin(1/x)'});
  add('sinh','hyper','sinh x',sinh,cosh,cosh,'cosh x','cosh x','ℝ','ℝ','奇函数','整个定义域上递增',all,{view:[-3,3,-4,4],inverse:'arsinh x',definition:'(eˣ − e⁻ˣ) / 2',note:'sinh 的导数是 cosh；与圆三角函数不同，cosh 的导数仍取正号。'});
  add('cosh','hyper','cosh x',cosh,sinh,sinh,'sinh x','sinh x','ℝ','[1, +∞)','偶函数','x < 0 时递减，x > 0 时递增',all,{view:[-3,3,-2,6],inverse:'arcosh x',inverseLimit:[0,Infinity],definition:'(eˣ + e⁻ˣ) / 2',note:'cosh²x − sinh²x = 1。取 x ≥ 0 的分支，才能得到反函数 arcosh。'});
  add('tanh','hyper','tanh x',tanh,x=>sech(x)**2,x=>log(cosh(x)),'sech²x','ln(cosh x)','ℝ','(−1, 1)','奇函数','整个定义域上递增',all,{inverse:'artanh x',definition:'sinh x / cosh x',horizontal:[-1,1]});
  add('sech','hyper','sech x',sech,x=>-sech(x)*tanh(x),x=>atan(sinh(x)),'−sech x · tanh x','arctan(sinh x)','ℝ','(0, 1]','偶函数','x < 0 时递增，x > 0 时递减',all,{inverse:'arsech x',inverseLimit:[0,Infinity],definition:'1 / cosh x',note:'取 x ≥ 0 的分支反演得到 arsech；sech 没有实数奇点。'});
  add('csch','hyper','csch x',csch,x=>-csch(x)*coth(x),x=>log(abs(tanh(x/2))),'−csch x · coth x','ln|tanh(x/2)|','ℝ ∖ {0}','ℝ ∖ {0}','奇函数','两个分支上分别递减',nonzero,{anchor:.5,interval:[.5,1.5],inverse:'arcsch x',definition:'1 / sinh x'});
  add('coth','hyper','coth x',coth,x=>-(csch(x)**2),x=>log(abs(sinh(x))),'−csch²x','ln|sinh x|','ℝ ∖ {0}','(−∞, −1) ∪ (1, +∞)','奇函数','两个分支上分别递减',nonzero,{anchor:.5,interval:[.5,1.5],inverse:'arcoth x',definition:'cosh x / sinh x',horizontal:[-1,1]});
  add('asinh','ihyper','arsinh x',asinh,x=>1/sqrt(1+x*x),x=>x*asinh(x)-sqrt(1+x*x),'1 / √(1 + x²)','x arsinh x − √(1 + x²)','ℝ','ℝ','奇函数','整个定义域上递增',all,{inverse:'sinh x',definition:'ln(x + √(x² + 1))',note:'arsinh 也记作 arcsinh 或 asinh；它是反函数，不是 1/sinh x。'});
  add('acosh','ihyper','arcosh x',acosh,x=>1/sqrt(x*x-1),x=>x*acosh(x)-sqrt(x*x-1),'1 / √(x² − 1)','x arcosh x − √(x² − 1)','[1, +∞)','[0, +∞)','非奇非偶','整个定义域上递增',positive,{view:[-.5,4.5,-1.5,4.5],point:1.7,anchor:1.2,interval:[1.2,2],edges:[1],inverse:'cosh x（x ≥ 0）',definition:'ln(x + √(x² − 1))'});
  add('atanh','ihyper','artanh x',atanh,x=>1/(1-x*x),x=>x*atanh(x)+.5*log(1-x*x),'1 / (1 − x²)','x artanh x + ½ ln(1 − x²)','(−1, 1)','ℝ','奇函数','整个定义域上递增',openInside,{view:[-1.8,1.8,-3.5,3.5],interval:[-.5,.5],inverse:'tanh x',definition:'½ ln((1 + x)/(1 − x))'});
  add('asech','ihyper','arsech x',asech,x=>-1/(x*sqrt(1-x*x)),x=>x*asech(x)+asin(x),'−1 / (x √(1 − x²))','x arsech x + arcsin x','(0, 1]','[0, +∞)','非奇非偶','整个定义域上递减',sechDomain,{view:[-.5,1.5,-1,4],point:.5,anchor:.25,interval:[.25,.75],edges:[1],inverse:'sech x（x ≥ 0）',definition:'arcosh(1/x)'});
  add('acsch','ihyper','arcsch x',acsch,x=>-1/(abs(x)*sqrt(1+x*x)),x=>x*acsch(x)+asinh(abs(x)),'−1 / (|x| √(1 + x²))','x arcsch x + arsinh|x|','ℝ ∖ {0}','ℝ ∖ {0}','奇函数','两个分支上分别递减',nonzero,{anchor:.5,interval:[.5,1.5],inverse:'csch x',definition:'arsinh(1/x)',note:'负半轴同样需要 |x|；ln|x + √(1 + x²)| 不能直接替代 arsinh|x|。'});
  add('acoth','ihyper','arcoth x',acoth,x=>1/(1-x*x),x=>x*acoth(x)+.5*log(x*x-1),'1 / (1 − x²)','x arcoth x + ½ ln(x² − 1)','(−∞, −1) ∪ (1, +∞)','ℝ ∖ {0}','奇函数','两个分支上分别递减',openOutside,{point:1.7,anchor:1.2,interval:[1.2,2],inverse:'coth x',definition:'½ ln((x + 1)/(x − 1))',note:'arcoth 的值域是 ℝ ∖ {0}。它和 artanh 的导数形式相同，但实数定义域不同。'});
  const kernelNote = '这里取参数 a = 1。可用 x = a·u（a > 0）还原带参数的积分；先确认根号与分母的定义域。';
  add('root-circle','kernels','1 / √(1 − x²)',x=>1/sqrt(1-x*x),x=>x/(1-x*x)**1.5,asin,'x / (1 − x²)³ᐟ²','arcsin x','(−1, 1)','[1, +∞)','偶函数','负半轴递减，正半轴递增',openInside,{view:[-1.6,1.6,-2,5],interval:[-.5,.5],note:kernelNote});
  add('root-plus','kernels','1 / √(1 + x²)',x=>1/sqrt(1+x*x),x=>-x/(1+x*x)**1.5,asinh,'−x / (1 + x²)³ᐟ²','arsinh x','ℝ','(0, 1]','偶函数','负半轴递增，正半轴递减',all,{note:kernelNote});
  add('root-minus','kernels','1 / √(x² − 1)',x=>1/sqrt(x*x-1),x=>-x/(x*x-1)**1.5,x=>Math.sign(x)*acosh(abs(x)),'−x / (x² − 1)³ᐟ²','ln|x + √(x² − 1)|','(−∞, −1) ∪ (1, +∞)','(0, +∞)','偶函数','负半轴递增，正半轴递减',openOutside,{point:1.7,anchor:1.2,interval:[1.2,2],note:'负半轴的原函数是 −arcosh(−x)，不能直接用 arcosh x。对数形式在两个分支上都成立。'});
  add('rational-plus','kernels','1 / (1 + x²)',x=>1/(1+x*x),x=>-2*x/(1+x*x)**2,atan,'−2x / (1 + x²)²','arctan x','ℝ','(0, 1]','偶函数','负半轴递增，正半轴递减',all,{note:kernelNote});
  const rationalBranch = x => x === -1 || x === 1 ? null : x < -1 ? -1 : x > 1 ? 1 : 0;
  add('rational-minus','kernels','1 / (1 − x²)',x=>1/(1-x*x),x=>2*x/(1-x*x)**2,x=>.5*log(abs((1+x)/(1-x))),'2x / (1 − x²)²','½ ln|(1 + x)/(1 − x)|','ℝ ∖ {−1, 1}','(−∞, 0) ∪ [1, +∞)','偶函数','各连续区间内，x < 0 递减，x > 0 递增',rationalBranch,{interval:[-.5,.5],note:'|x| < 1 时原函数为 artanh x；|x| > 1 时为 arcoth x。不能跨越 x = ±1。'});
  add('sec-square','kernels','sec²x',x=>sec(x)**2,x=>2*sec(x)**2*tan(x),tan,'2 sec²x · tan x','tan x','x ≠ π/2 + kπ','[1, +∞)','偶函数','周期 π；分段单调',cosBranch,{view:trigView});
  const byId = Object.fromEntries(functions.map(f=>[f.id,f]));
  function value(fn, mode, x, C = 0) {
    if (!Number.isFinite(x) || fn.branch(x) === null) return NaN;
    const result = mode === 'derivative' ? fn.d(x) : mode === 'integral' ? fn.F(x)+C : fn.f(x);
    return Number.isFinite(result) ? result : NaN;
  }
  function integral(fn, a, b) {
    if (!Number.isFinite(a) || !Number.isFinite(b)) return {ok:false, message:'请输入有限的积分上下限。'};
    const branch = fn.branch(a);
    if (branch === null || fn.branch(b) !== branch || !Number.isFinite(fn.f(a)) || !Number.isFinite(fn.f(b))) {
      return {ok:false,message:'区间含定义域外的点或奇点，请把上下限放在同一连续分支内。'};
    }
    // The branch identifiers above encode every domain break, including periodic poles.
    const result = fn.F(b) - fn.F(a);
    return Number.isFinite(result) ? {ok:true,value:result} : {ok:false,message:'该区间无法计算有限的定积分。'};
  }
  const circlePoint = t => ({x:cos(t),y:sin(t)});
  const hyperbolaPoint = t => ({x:cosh(t),y:sinh(t)});
  const api = {groups,functions,byId,value,integral,circlePoint,hyperbolaPoint};
  root.FunctionLab = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
