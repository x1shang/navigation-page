/* 点球落点预测 — 在浏览器里跑论文冻结的 P1 混合密度网络。
 *
 * 三点坐标 -> 11 维物理特征 -> 标准化 -> 3 层 MLP（LayerNorm + GELU）
 *          -> 3 个二维高斯分量 + 校准尺度 -> 球门平面上的混合密度热力图。
 *
 * 权重与数值口径见 model.json 与页内「方法」一节。
 */
(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /* ───────────────────────────── 常量 ───────────────────────────── */
  const C = {
    r0: [0, 0, 0.11],
    dt: 0.05,
    gravity: 9.81,
    goalX: 11,
    halfWidth: 3.66,
    goalHeight: 2.44,
  };
  const CASE_POINTS = [[0, 0, 0.11], [1.5, 0.45, 0.44], [2.96, 0.86, 0.74]];

  const FEATURE_INFO = [
    ['v_x', 'm/s'], ['v_y', 'm/s'], ['v_z', 'm/s'],
    ['a_x', 'm/s²'], ['a_y', 'm/s²'], ['a_z', 'm/s²'],
    ['|v|', 'm/s'], ['a∥', 'm/s²'], ['a⊥', 'm/s²'],
    ['t 过门', 'ms'], ['σ', 'mm'],
  ];

  const state = {
    meta: null,
    models: new Map(),
    points: CASE_POINTS.map((p) => p.slice()),
    sigma: 5,
    seed: '11',
    hover: null,
    last: null,
  };

  /* ───────────────────────────── 数学 ───────────────────────────── */
  const SQRT1_2 = 0.7071067811865476;
  const INV_2PI = 1 / (2 * Math.PI);

  function erf(x) {
    const s = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * ax);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
    return s * y;
  }
  const normCdf = (x) => 0.5 * (1 + erf(x * SQRT1_2));
  const gelu = (x) => 0.5 * x * (1 + erf(x * SQRT1_2));
  const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

  function timeToGoal(vx, ax) {
    if (Math.abs(ax) < 1e-9) return C.goalX / Math.max(vx, 1e-9);
    const disc = vx * vx + 22 * ax;
    if (disc < 0) return NaN;
    const sq = Math.sqrt(disc);
    const t1 = (-vx + sq) / ax;
    const t2 = (-vx - sq) / ax;
    let best = Infinity;
    if (t1 > 0.1) best = Math.min(best, t1);
    if (t2 > 0.1) best = Math.min(best, t2);
    return Number.isFinite(best) ? best : NaN;
  }

  /* 三点 → 11 维特征（对应论文 common.py: physics_features + phys_reduced 列） */
  function buildFeatures(points, sigmaM) {
    const [p0, p1, p2] = points;
    const dt2 = C.dt * C.dt;
    const v = [
      (p2[0] - p0[0]) / (2 * C.dt),
      (p2[1] - p0[1]) / (2 * C.dt),
      (p2[2] - p0[2]) / (2 * C.dt),
    ];
    const a = [
      (p2[0] - 2 * p1[0] + p0[0]) / dt2,
      (p2[1] - 2 * p1[1] + p0[1]) / dt2,
      (p2[2] - 2 * p1[2] + p0[2]) / dt2,
    ];
    const aero = [a[0], a[1], a[2] + C.gravity];
    const speed = Math.hypot(v[0], v[1], v[2]);
    const inv = 1 / Math.max(speed, 1e-9);
    const vh = [v[0] * inv, v[1] * inv, v[2] * inv];
    const aPar = aero[0] * vh[0] + aero[1] * vh[1] + aero[2] * vh[2];
    const aPerp = Math.hypot(
      aero[0] - aPar * vh[0],
      aero[1] - aPar * vh[1],
      aero[2] - aPar * vh[2],
    );
    const tGoal = timeToGoal(v[0] - a[0] * C.dt, a[0]);
    return {
      x: [v[0], v[1], v[2], a[0], a[1], a[2], speed, aPar, aPerp, tGoal, sigmaM],
      display: [v[0], v[1], v[2], a[0], a[1], a[2], speed, aPar, aPerp, tGoal * 1000, sigmaM * 1000],
      v, a, aero, speed, aPar, aPerp, tGoal,
    };
  }

  /* 恒加速度解析基线 */
  function baseline(points) {
    const [p0, p1, p2] = points;
    const dt2 = C.dt * C.dt;
    const a = [
      (p2[0] - 2 * p1[0] + p0[0]) / dt2,
      (p2[1] - 2 * p1[1] + p0[1]) / dt2,
      (p2[2] - 2 * p1[2] + p0[2]) / dt2,
    ];
    const v = [
      (p1[0] - p0[0]) / C.dt - 0.5 * a[0] * C.dt,
      (p1[1] - p0[1]) / C.dt - 0.5 * a[1] * C.dt,
      (p1[2] - p0[2]) / C.dt - 0.5 * a[2] * C.dt,
    ];
    const t = timeToGoal(v[0], a[0]);
    if (!Number.isFinite(t)) return null;
    const y = v[1] * t + 0.5 * a[1] * t * t;
    const z = C.r0[2] + v[2] * t + 0.5 * a[2] * t * t;
    if (Math.abs(y) > 15 || Math.abs(z) > 15) return null;
    return [y, z];
  }

  /* ───────────────────────────── 模型 ───────────────────────────── */
  async function loadModel() {
    const meta = await fetch('./model.json', { cache: 'force-cache' }).then((r) => {
      if (!r.ok) throw new Error('model.json ' + r.status);
      return r.json();
    });
    const buf = await fetch('./' + meta.weightsFile, { cache: 'force-cache' }).then((r) => {
      if (!r.ok) throw new Error(meta.weightsFile + ' ' + r.status);
      return r.arrayBuffer();
    });
    state.meta = meta;
    const H = meta.arch.hidden;
    for (const spec of meta.seeds) {
      const tensors = {};
      for (const t of meta.tensors) {
        tensors[t.name] = new Float32Array(buf, spec.byteOffset + t.offset, t.count);
      }
      const block = (lin, ln, rows, cols) => ({
        w: tensors[lin + '.weight'], b: tensors[lin + '.bias'],
        g: tensors[ln + '.weight'], beta: tensors[ln + '.bias'], rows, cols,
      });
      state.models.set(String(spec.seed), {
        seed: spec.seed,
        stats: spec.stats,
        cal: spec.calibration,
        bufA: new Float64Array(H),
        bufB: new Float64Array(H),
        out: new Float64Array(6 * meta.arch.components),
        blocks: [
          block('net.0', 'net.1', H, 11),
          block('net.3', 'net.4', H, H),
          block('net.6', 'net.7', H, H),
        ],
        head: { w: tensors['net.9.weight'], b: tensors['net.9.bias'], rows: 18, cols: H },
      });
    }
  }

  /* 前向：返回 18 个原始输出。两块交换缓冲，写入目标永远不是当前输入。 */
  function forward(model, x) {
    const out = model.out;
    const bufs = [model.bufA, model.bufB];
    let cur = x;
    let wi = 0;
    for (const blk of model.blocks) {
      const { w, b: bias, g, beta, rows, cols } = blk;
      const dst = bufs[wi];
      for (let i = 0; i < rows; i++) {
        let s = bias[i];
        const off = i * cols;
        for (let j = 0; j < cols; j++) s += w[off + j] * cur[j];
        dst[i] = s;
      }
      // LayerNorm + GELU，原地
      let mean = 0;
      for (let i = 0; i < rows; i++) mean += dst[i];
      mean /= rows;
      let varr = 0;
      for (let i = 0; i < rows; i++) { const d = dst[i] - mean; varr += d * d; }
      const inv = 1 / Math.sqrt(varr / rows + 1e-5);
      for (let i = 0; i < rows; i++) dst[i] = gelu((dst[i] - mean) * inv * g[i] + beta[i]);
      cur = dst;
      wi = 1 - wi;
    }
    const head = model.head;
    for (let i = 0; i < head.rows; i++) {
      let s = head.b[i];
      const off = i * head.cols;
      for (let j = 0; j < head.cols; j++) s += head.w[off + j] * cur[j];
      out[i] = s;
    }
    return out;
  }

  /* 原始输出 → 米制分量（含校准尺度） */
  function componentsFor(model, raw, sigmaMm) {
    const cal = model.cal[String(sigmaMm)];
    if (!cal) throw new Error('缺少 ' + sigmaMm + ' mm 校准参数');
    const scale = cal.scale;
    const { ym, ys } = model.stats;
    const l0 = raw[0], l1 = raw[6], l2 = raw[12];
    const mx = Math.max(l0, l1, l2);
    const e0 = Math.exp(l0 - mx), e1 = Math.exp(l1 - mx), e2 = Math.exp(l2 - mx);
    const sum = e0 + e1 + e2;
    const ws = [e0 / sum, e1 / sum, e2 / sum];
    const out = { p: [], my: [], mz: [], L00: [], L10: [], L11: [] };
    for (let k = 0; k < 3; k++) {
      const o = k * 6;
      const s0 = softplus(raw[o + 3]) + 1e-3;
      const s1 = softplus(raw[o + 4]) + 1e-3;
      const c = 0.995 * Math.tanh(raw[o + 5]);
      out.p.push(ws[k]);
      out.my.push(raw[o + 1] * ys[0] + ym[0]);
      out.mz.push(raw[o + 2] * ys[1] + ym[1]);
      out.L00.push(s0 * scale * ys[0]);
      out.L10.push(c * s1 * scale * ys[1]);
      out.L11.push(s1 * Math.sqrt(1 - c * c) * scale * ys[1]);
    }
    return out;
  }

  /* 多种子等权合并成 3K 个分量 */
  function mergeComponents(list) {
    if (list.length === 1) return list[0];
    const out = { p: [], my: [], mz: [], L00: [], L10: [], L11: [] };
    const n = list.length;
    for (const c of list) {
      for (let k = 0; k < 3; k++) {
        out.p.push(c.p[k] / n);
        out.my.push(c.my[k]);
        out.mz.push(c.mz[k]);
        out.L00.push(c.L00[k]);
        out.L10.push(c.L10[k]);
        out.L11.push(c.L11[k]);
      }
    }
    return out;
  }

  function prepDensity(comps) {
    const K = comps.p.length;
    const d = {
      K,
      p: Float64Array.from(comps.p),
      my: Float64Array.from(comps.my),
      mz: Float64Array.from(comps.mz),
      i00: new Float64Array(K), i01: new Float64Array(K), i11: new Float64Array(K),
      norm: new Float64Array(K),
    };
    for (let k = 0; k < K; k++) {
      const l00 = comps.L00[k], l10 = comps.L10[k], l11 = comps.L11[k];
      const det = l00 * l11;
      const inv = 1 / (det * det);
      d.i00[k] = (l10 * l10 + l11 * l11) * inv;
      d.i01[k] = -l00 * l10 * inv;
      d.i11[k] = l00 * l00 * inv;
      d.norm[k] = d.p[k] * INV_2PI / Math.abs(det);
    }
    return d;
  }

  function densityAt(d, y, z) {
    let s = 0;
    for (let k = 0; k < d.K; k++) {
      const dy = y - d.my[k];
      const dz = z - d.mz[k];
      const q = d.i00[k] * dy * dy + 2 * d.i01[k] * dy * dz + d.i11[k] * dz * dz;
      if (q < 80) s += d.norm[k] * Math.exp(-0.5 * q);
    }
    return s;
  }

  /* 视窗：先在宽窗口上粗算一遍密度，再裁到 99% 概率区域的包围盒。
     这样不管分布是窄是宽，图都填得比较满。 */
  function probeWindow(comps) {
    let yl = Infinity, yh = -Infinity, zl = Infinity, zh = -Infinity;
    for (let k = 0; k < comps.p.length; k++) {
      const sy = Math.abs(comps.L00[k]);
      const sz = Math.hypot(comps.L10[k], comps.L11[k]);
      yl = Math.min(yl, comps.my[k] - 6 * sy);
      yh = Math.max(yh, comps.my[k] + 6 * sy);
      zl = Math.min(zl, comps.mz[k] - 6 * sz);
      zh = Math.max(zh, comps.mz[k] + 6 * sz);
    }
    if (!Number.isFinite(yl)) return { yl: -C.halfWidth, yh: C.halfWidth, zl: 0, zh: C.goalHeight };
    const padY = 0.04 * (yh - yl), padZ = 0.04 * (zh - zl);
    return { yl: yl - padY, yh: yh + padY, zl: zl - padZ, zh: zh + padZ };
  }

  function windowFor(comps, aspect, dens) {
    const probe = probeWindow(comps);
    const N = 200;
    const pf = buildField(dens, probe, N, N);
    const t = levelForMass(pf, 0.99);
    let yl = Infinity, yh = -Infinity, zl = Infinity, zh = -Infinity;
    const dx = (probe.yh - probe.yl) / (N - 1);
    const dz = (probe.zh - probe.zl) / (N - 1);
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (pf.d[j * N + i] < t) continue;
        const y = probe.yl + i * dx;
        const z = probe.zl + j * dz;
        if (y < yl) yl = y;
        if (y > yh) yh = y;
        if (z < zl) zl = z;
        if (z > zh) zh = z;
      }
    }
    if (!Number.isFinite(yl) || yh - yl < 1e-3 || zh - zl < 1e-3) {
      return { yl: probe.yl, yh: probe.yh, zl: probe.zl, zh: probe.zh };
    }
    const wy = Math.max(yh - yl, 0.9);
    const wz = Math.max(zh - zl, 0.7);
    const cy = (yl + yh) / 2, cz = (zl + zh) / 2;
    yl = cy - wy / 2; yh = cy + wy / 2;
    zl = cz - wz / 2; zh = cz + wz / 2;
    const padY = 0.10 * (yh - yl), padZ = 0.10 * (zh - zl);
    yl -= padY; yh += padY; zl -= padZ; zh += padZ;
    const W = yh - yl, Z = zh - zl;
    if (W / Z < aspect) {
      const need = Z * aspect - W;
      yl -= need / 2; yh += need / 2;
    } else {
      const need = W / aspect - Z;
      zl -= need / 2; zh += need / 2;
    }
    return { yl, yh, zl, zh };
  }

  function buildField(d, win, nx, ny) {
    const dx = (win.yh - win.yl) / (nx - 1);
    const dz = (win.zh - win.zl) / (ny - 1);
    const field = new Float32Array(nx * ny);
    const K = d.K;
    const bK = new Float64Array(K), cK = new Float64Array(K);
    let max = 0, maxI = 0;
    for (let j = 0; j < ny; j++) {
      const z = win.zl + j * dz;
      for (let k = 0; k < K; k++) {
        const dzK = z - d.mz[k];
        bK[k] = 2 * d.i01[k] * dzK;
        cK[k] = d.i11[k] * dzK * dzK;
      }
      const row = j * nx;
      for (let i = 0; i < nx; i++) {
        const y = win.yl + i * dx;
        let s = 0;
        for (let k = 0; k < K; k++) {
          const dy = y - d.my[k];
          const q = d.i00[k] * dy * dy + bK[k] * dy + cK[k];
          if (q < 80) s += d.norm[k] * Math.exp(-0.5 * q);
        }
        field[row + i] = s;
        if (s > max) { max = s; maxI = row + i; }
      }
    }
    return { d: field, nx, ny, dx, dz, max, maxI };
  }

  function levelForMass(field, target) {
    const { d, max } = field;
    const NB = 2048;
    const binMass = new Float64Array(NB);
    let total = 0;
    const inv = max > 0 ? NB / max : 0;
    for (let i = 0; i < d.length; i++) {
      const v = d[i];
      total += v;
      let b = (v * inv) | 0;
      if (b >= NB) b = NB - 1;
      if (b < 0) b = 0;
      binMass[b] += v;
    }
    if (!(total > 0)) return 0;
    const want = target * total;
    let acc = 0;
    for (let b = NB - 1; b >= 0; b--) {
      if (acc + binMass[b] >= want) {
        const need = want - acc;
        const lo = b / inv, hi = (b + 1) / inv;
        const frac = binMass[b] > 0 ? need / binMass[b] : 1;
        return hi - frac * (hi - lo);
      }
      acc += binMass[b];
    }
    return 0;
  }

  /* 行进方格：共享边用统一编号，保证相邻单元接得上 */
  const MS_TABLE = {
    1: [0, 3], 2: [0, 1], 3: [3, 1], 4: [1, 2], 5: [3, 2, 1, 0],
    6: [0, 2], 7: [3, 2], 8: [2, 3], 9: [0, 2], 10: [1, 2, 3, 0],
    11: [1, 2], 12: [3, 1], 13: [0, 1], 14: [0, 3],
  };

  function marchingSquares(field, level) {
    const { d, nx, ny } = field;
    const pts = [];
    const index = new Map();
    const segs = [];
    const getPoint = (key, x, y) => {
      let idx = index.get(key);
      if (idx === undefined) {
        idx = pts.length / 2;
        pts.push(x, y);
        index.set(key, idx);
      }
      return idx;
    };
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const i0 = j * nx + i;
        const v00 = d[i0], v10 = d[i0 + 1], v11 = d[i0 + nx + 1], v01 = d[i0 + nx];
        let code = 0;
        if (v00 >= level) code |= 1;
        if (v10 >= level) code |= 2;
        if (v11 >= level) code |= 4;
        if (v01 >= level) code |= 8;
        if (code === 0 || code === 15) continue;
        const seg = MS_TABLE[code];
        if (!seg) continue;
        const edgePoint = (e) => {
          switch (e) {
            case 0: { const f = (level - v00) / ((v10 - v00) || 1e-12); return getPoint((j * nx + i) * 2, i + f, j); }
            case 1: { const f = (level - v10) / ((v11 - v10) || 1e-12); return getPoint((j * nx + i + 1) * 2 + 1, i + 1, j + f); }
            case 2: { const f = (level - v01) / ((v11 - v01) || 1e-12); return getPoint(((j + 1) * nx + i) * 2, i + f, j + 1); }
            default: { const f = (level - v00) / ((v01 - v00) || 1e-12); return getPoint((j * nx + i) * 2 + 1, i, j + f); }
          }
        };
        for (let s = 0; s < seg.length; s += 2) {
          const a = edgePoint(seg[s]);
          const b = edgePoint(seg[s + 1]);
          if (a !== b) segs.push([a, b]);
        }
      }
    }
    return { pts: Float64Array.from(pts), segs, nx, ny };
  }

  function contourArea(contour, field) {
    const { segs, pts } = contour;
    if (!segs.length) return 0;
    const adj = new Map();
    const add = (p, s) => {
      const list = adj.get(p);
      if (list) list.push(s); else adj.set(p, [s]);
    };
    segs.forEach((s, i) => { add(s[0], i); add(s[1], i); });
    const used = new Uint8Array(segs.length);
    let area = 0;
    for (let i = 0; i < segs.length; i++) {
      if (used[i]) continue;
      used[i] = 1;
      const start = segs[i][0];
      let cur = segs[i][1];
      const poly = [start, cur];
      let guard = 0;
      while (cur !== start && guard++ < segs.length + 2) {
        const list = adj.get(cur);
        if (!list) break;
        let next = -1;
        for (let s = 0; s < list.length; s++) { if (!used[list[s]]) { next = list[s]; break; } }
        if (next < 0) break;
        used[next] = 1;
        cur = segs[next][0] === cur ? segs[next][1] : segs[next][0];
        poly.push(cur);
      }
      let acc = 0;
      for (let k = 0; k < poly.length; k++) {
        const a = poly[k], b = poly[(k + 1) % poly.length];
        acc += pts[a * 2] * pts[b * 2 + 1] - pts[b * 2] * pts[a * 2 + 1];
      }
      area += acc / 2;
    }
    return Math.abs(area) * field.dx * field.dz;
  }

  /* 门户矩形上的混合高斯积分（每分量 64 点 Gauss–Legendre） */
  const GL = (() => {
    const n = 64;
    const x = new Float64Array(n), w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
      let pp = 0;
      for (let it = 0; it < 100; it++) {
        let p1 = 1, p2 = 0;
        for (let j = 0; j < n; j++) {
          const p3 = p2;
          p2 = p1;
          p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1);
        }
        pp = (n * (z * p1 - p2)) / (z * z - 1);
        const dz = p1 / pp;
        z -= dz;
        if (Math.abs(dz) < 1e-14) break;
      }
      x[i] = z;
      w[i] = 2 / ((1 - z * z) * pp * pp);
    }
    return { x, w };
  })();

  function probInside(comps, yl, yh, zl, zh) {
    const half = 0.5 * (zh - zl);
    const mid = 0.5 * (zh + zl);
    let total = 0;
    for (let k = 0; k < comps.p.length; k++) {
      const l00 = comps.L00[k], l10 = comps.L10[k], l11 = comps.L11[k];
      const sy = Math.abs(l00);
      const sz = Math.hypot(l10, l11);
      const rho = l10 / Math.max(sz, 1e-12);
      const s = sy * Math.sqrt(Math.max(1 - rho * rho, 1e-12));
      const my = comps.my[k], mz = comps.mz[k];
      let acc = 0;
      for (let i = 0; i < GL.x.length; i++) {
        const z = mid + half * GL.x[i];
        const shift = (rho * sy * (z - mz)) / sz;
        const up = normCdf((yh - my - shift) / s);
        const lo = normCdf((yl - my - shift) / s);
        const pdf = Math.exp(-0.5 * ((z - mz) / sz) ** 2) / (Math.sqrt(2 * Math.PI) * sz);
        acc += GL.w[i] * pdf * (up - lo);
      }
      total += comps.p[k] * acc * half;
    }
    return Math.min(1, Math.max(0, total));
  }

  /* 众数：网格最大值 + 对数密度二次插值，再用牛顿法在解析密度上收敛 */
  function refineMode(field, win, dens) {
    const { d, nx, ny, dx, dz, max, maxI } = field;
    if (!(max > 0)) return null;
    const bi = maxI % nx;
    const bj = (maxI - bi) / nx;
    let yi = bi, zj = bj;
    if (bi > 0 && bi < nx - 1 && bj > 0 && bj < ny - 1) {
      const L = (i, j) => Math.log(Math.max(d[j * nx + i], 1e-30));
      const c = L(bi, bj);
      const gx = (L(bi + 1, bj) - L(bi - 1, bj)) / 2;
      const gz = (L(bi, bj + 1) - L(bi, bj - 1)) / 2;
      const hx = L(bi + 1, bj) - 2 * c + L(bi - 1, bj);
      const hz = L(bi, bj + 1) - 2 * c + L(bi, bj - 1);
      if (hx < 0) yi = bi - Math.max(-0.5, Math.min(0.5, gx / hx));
      if (hz < 0) zj = bj - Math.max(-0.5, Math.min(0.5, gz / hz));
    }
    return polishMode(dens, win.yl + yi * dx, win.zl + zj * dz);
  }

  function polishMode(dens, y0, z0) {
    let y = y0, z = z0;
    const h = 0.002;
    const L = (yy, zz) => Math.log(Math.max(densityAt(dens, yy, zz), 1e-300));
    for (let it = 0; it < 40; it++) {
      const c = L(y, z);
      const a1 = L(y + h, z), a0 = L(y - h, z);
      const b1 = L(y, z + h), b0 = L(y, z - h);
      const gy = (a1 - a0) / (2 * h), gz = (b1 - b0) / (2 * h);
      const hy = (a1 - 2 * c + a0) / (h * h), hz = (b1 - 2 * c + b0) / (h * h);
      let dy = hy < 0 ? -gy / hy : 0;
      let dz = hz < 0 ? -gz / hz : 0;
      dy = Math.max(-0.05, Math.min(0.05, dy));
      dz = Math.max(-0.05, Math.min(0.05, dz));
      y += dy; z += dz;
      if (Math.abs(dy) + Math.abs(dz) < 1e-7) break;
    }
    return [y, z];
  }

  /* ───────────────────────────── 计算 ───────────────────────────── */
  function compute(aspect) {
    const feats = buildFeatures(state.points, state.sigma / 1000);
    const seedKeys = state.seed === 'avg' ? ['11', '22', '33'] : [state.seed];
    const perSeed = [];
    let maxZ = 0;
    let zscores = null;
    for (const key of seedKeys) {
      const model = state.models.get(key);
      if (!model) throw new Error('模型未载入');
      const { xm, xs } = model.stats;
      const std = new Float64Array(11);
      const zs = new Array(11);
      for (let i = 0; i < 11; i++) {
        let z = (feats.x[i] - xm[i]) / xs[i];
        if (!Number.isFinite(z)) z = 0;
        std[i] = z;
        zs[i] = z;
        if (i < 10) maxZ = Math.max(maxZ, Math.abs(z));
      }
      if (!zscores) zscores = zs;
      const raw = forward(model, std);
      perSeed.push(componentsFor(model, raw, state.sigma));
    }
    const comps = mergeComponents(perSeed);
    const dens = prepDensity(comps);
    const win = windowFor(comps, aspect, dens);

    const nx = Math.max(240, Math.min(480, Math.round(canvasCss.width * 1.15)));
    const ny = Math.max(200, Math.round(nx / aspect));
    const field = buildField(dens, win, nx, ny);

    let m50 = 0, m95 = 0;
    for (const key of seedKeys) {
      const cal = state.models.get(key).cal[String(state.sigma)];
      m50 += cal.mass50 / seedKeys.length;
      m95 += cal.mass95 / seedKeys.length;
    }
    const level50 = levelForMass(field, m50);
    const level95 = levelForMass(field, m95);
    const c50 = marchingSquares(field, level50);
    const c95 = marchingSquares(field, level95);
    const a50 = contourArea(c50, field);
    const a95 = contourArea(c95, field);

    let my = 0, mz = 0;
    for (let k = 0; k < comps.p.length; k++) {
      my += comps.p[k] * comps.my[k];
      mz += comps.p[k] * comps.mz[k];
    }

    return {
      feats, zscores, maxZ, win, field, comps, dens,
      level50, level95,
      contours: { c50, c95 },
      stats: {
        mode: refineMode(field, win, dens),
        mean: [my, mz],
        pGoal: probInside(comps, -C.halfWidth, C.halfWidth, 0, C.goalHeight),
        a50, a95,
        base: baseline(state.points),
        m50, m95,
      },
    };
  }

  /* ───────────────────────────── 绘制 ───────────────────────────── */
  const canvas = $('#heat');
  const ctx = canvas.getContext('2d');
  const canvasWrap = $('#canvas-wrap');
  const canvasCss = { width: 4, height: 3 };
  const base = document.createElement('canvas');
  const baseCtx = base.getContext('2d');
  let off = document.createElement('canvas');
  let offCtx = off.getContext('2d');
  let imageData = null;
  let lut = null;
  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);

  function buildLut() {
    const stops = [
      [0.00, 10, 8, 24, 0],
      [0.07, 30, 19, 62, 34],
      [0.20, 66, 30, 104, 92],
      [0.38, 116, 40, 126, 146],
      [0.56, 168, 52, 122, 184],
      [0.72, 216, 92, 96, 212],
      [0.86, 244, 152, 82, 232],
      [0.95, 253, 208, 122, 242],
      [1.00, 255, 244, 214, 248],
    ];
    const out = new Uint8ClampedArray(256 * 4);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let a = stops[0], b = stops[stops.length - 1];
      for (let s = 0; s < stops.length - 1; s++) {
        if (t >= stops[s][0] && t <= stops[s + 1][0]) { a = stops[s]; b = stops[s + 1]; break; }
      }
      const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
      out[i * 4] = a[1] + (b[1] - a[1]) * f;
      out[i * 4 + 1] = a[2] + (b[2] - a[2]) * f;
      out[i * 4 + 2] = a[3] + (b[3] - a[3]) * f;
      out[i * 4 + 3] = a[4] + (b[4] - a[4]) * f;
    }
    return out;
  }

  function resizeCanvas() {
    const rect = canvasWrap.getBoundingClientRect();
    const k = dpr();
    canvasCss.width = Math.max(1, rect.width);
    canvasCss.height = Math.max(1, rect.height);
    canvas.width = Math.round(canvasCss.width * k);
    canvas.height = Math.round(canvasCss.height * k);
    base.width = canvas.width;
    base.height = canvas.height;
    ctx.setTransform(k, 0, 0, k, 0, 0);
    baseCtx.setTransform(k, 0, 0, k, 0, 0);
  }

  function niceStep(span, target) {
    const raw = span / Math.max(1, target);
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / pow;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * pow;
  }

  const fmt = (v, digits = 2) => (Number.isFinite(v) ? v.toFixed(digits) : '—');
  const fmtInput = (v) => (v.toFixed(3).endsWith('0') ? v.toFixed(2) : v.toFixed(3));

  /* 底图：热力图 + 网格 + 门框 + 等值线 + 刻度 */
  function renderBase(res) {
    const W = canvasCss.width;
    const H = canvasCss.height;
    const { win, field } = res;
    const toX = (y) => ((y - win.yl) / (win.yh - win.yl)) * W;
    const toY = (z) => H - ((z - win.zl) / (win.zh - win.zl)) * H;
    const g = baseCtx;

    g.save();
    g.fillStyle = '#07060d';
    g.fillRect(0, 0, W, H);

    if (field.max > 0) {
      if (!lut) lut = buildLut();
      if (!imageData || imageData.width !== field.nx || imageData.height !== field.ny) {
        off.width = field.nx;
        off.height = field.ny;
        offCtx = off.getContext('2d');
        imageData = offCtx.createImageData(field.nx, field.ny);
      }
      const data = imageData.data;
      const inv = 1 / field.max;
      for (let j = 0; j < field.ny; j++) {
        const src = j * field.nx;
        const dstRow = (field.ny - 1 - j) * field.nx;
        for (let i = 0; i < field.nx; i++) {
          let t = field.d[src + i] * inv;
          if (!(t > 0)) t = 0; else if (t > 1) t = 1;
          t = Math.pow(t, 0.72);
          const li = (t * 255) | 0;
          const o = (dstRow + i) * 4;
          const q = li * 4;
          data[o] = lut[q];
          data[o + 1] = lut[q + 1];
          data[o + 2] = lut[q + 2];
          data[o + 3] = lut[q + 3];
        }
      }
      offCtx.putImageData(imageData, 0, 0);
      g.imageSmoothingEnabled = true;
      g.imageSmoothingQuality = 'high';
      g.drawImage(off, 0, 0, W, H);
    }

    /* 网格线 */
    const step = niceStep(win.yh - win.yl, 6);
    const zstep = niceStep(win.zh - win.zl, 4);
    g.lineWidth = 1;
    g.strokeStyle = 'rgba(255,255,255,0.07)';
    g.beginPath();
    for (let y = Math.ceil(win.yl / step) * step; y <= win.yh; y += step) {
      const x = Math.round(toX(y)) + 0.5;
      g.moveTo(x, 0); g.lineTo(x, H);
    }
    for (let z = Math.ceil(win.zl / zstep) * zstep; z <= win.zh; z += zstep) {
      const y = Math.round(toY(z)) + 0.5;
      g.moveTo(0, y); g.lineTo(W, y);
    }
    g.stroke();

    /* 门框 */
    const gx0 = toX(-C.halfWidth), gx1 = toX(C.halfWidth);
    const gy0 = toY(C.goalHeight), gy1 = toY(0);
    g.fillStyle = 'rgba(126,226,168,0.055)';
    g.fillRect(gx0, gy0, gx1 - gx0, gy1 - gy0);
    g.save();
    g.setLineDash([6, 5]);
    g.lineWidth = 1.6;
    g.strokeStyle = 'rgba(126,226,168,0.85)';
    g.strokeRect(gx0, gy0, gx1 - gx0, gy1 - gy0);
    g.restore();

    /* 等值线 */
    const strokeContour = (contour, color, width) => {
      if (!contour.segs.length) return;
      const sx = W / (field.nx - 1);
      const sy = H / (field.ny - 1);
      const pts = contour.pts;
      g.beginPath();
      for (const seg of contour.segs) {
        const a = seg[0], b = seg[1];
        g.moveTo(pts[a * 2] * sx, H - pts[a * 2 + 1] * sy);
        g.lineTo(pts[b * 2] * sx, H - pts[b * 2 + 1] * sy);
      }
      g.lineWidth = width;
      g.strokeStyle = color;
      g.lineJoin = 'round';
      g.stroke();
    };
    strokeContour(res.contours.c95, 'rgba(70,224,208,0.85)', 1.5);
    strokeContour(res.contours.c50, 'rgba(255,255,255,0.95)', 1.9);

    /* 刻度 */
    g.font = '10px ui-monospace,SFMono-Regular,Menlo,monospace';
    g.fillStyle = 'rgba(200,195,220,0.55)';
    g.textAlign = 'center';
    g.textBaseline = 'bottom';
    for (let y = Math.ceil(win.yl / step) * step; y <= win.yh; y += step) {
      const x = toX(y);
      if (x < 16 || x > W - 44) continue;
      g.fillText(String(Math.round(y * 10) / 10), x, H - 5);
    }
    g.textAlign = 'left';
    g.textBaseline = 'middle';
    for (let z = Math.ceil(win.zl / zstep) * zstep; z <= win.zh; z += zstep) {
      const y = toY(z);
      if (y < 12 || y > H - 20) continue;
      g.fillText(String(Math.round(z * 10) / 10), 5, y);
    }
    g.fillStyle = 'rgba(200,195,220,0.45)';
    g.textAlign = 'right';
    g.textBaseline = 'bottom';
    g.fillText('Y (m) →', W - 6, H - 5);
    g.restore();
  }

  /* 上层：标记 + 悬停十字 */
  function drawGlyph(g, x, y, kind) {
    if (kind === 'plus') {
      g.moveTo(x - 7, y); g.lineTo(x + 7, y);
      g.moveTo(x, y - 7); g.lineTo(x, y + 7);
    } else if (kind === 'cross') {
      g.moveTo(x - 5.5, y - 5.5); g.lineTo(x + 5.5, y + 5.5);
      g.moveTo(x + 5.5, y - 5.5); g.lineTo(x - 5.5, y + 5.5);
    } else {
      g.moveTo(x, y - 6.5); g.lineTo(x + 6.5, y); g.lineTo(x, y + 6.5); g.lineTo(x - 6.5, y);
      g.closePath();
    }
  }

  function paint() {
    const res = state.last;
    if (!res) return;
    const W = canvasCss.width;
    const H = canvasCss.height;
    const { win } = res;
    const toX = (y) => ((y - win.yl) / (win.yh - win.yl)) * W;
    const toY = (z) => H - ((z - win.zl) / (win.zh - win.zl)) * H;

    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(base, 0, 0, W, H);

    if (state.hover) {
      const x = toX(state.hover.y);
      const y = toY(state.hover.z);
      ctx.save();
      ctx.strokeStyle = 'rgba(201,182,255,0.5)';
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0); ctx.lineTo(x, H);
      ctx.moveTo(0, y); ctx.lineTo(W, y);
      ctx.stroke();
      ctx.restore();
    }

    const marker = (pt, kind, color, label) => {
      if (!pt) return;
      const x = toX(pt[0]), y = toY(pt[1]);
      if (x < -20 || x > W + 20 || y < -20 || y > H + 20) return;
      ctx.save();
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'rgba(8,8,14,0.55)';
      ctx.lineWidth = 4.4;
      ctx.beginPath();
      drawGlyph(ctx, x, y, kind);
      ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.7;
      ctx.beginPath();
      drawGlyph(ctx, x, y, kind);
      ctx.stroke();

      /* 标签：四个候选位置，避开已放置的标签与画布边缘 */
      ctx.font = '10.5px system-ui,"Microsoft YaHei",sans-serif';
      const tw = ctx.measureText(label).width;
      const cand = [
        [x + 10, y - 16], [x - tw - 12, y - 16], [x + 10, y + 6], [x - tw - 12, y + 6],
      ];
      let place = null;
      for (const [cx, cy] of cand) {
        if (cx < 2 || cx + tw + 5 > W - 2 || cy < 1 || cy + 14 > H - 1) continue;
        const rect = { x: cx, y: cy, w: tw + 5, h: 14 };
        if (placed.some((p) => !(rect.x + rect.w < p.x || rect.x > p.x + p.w || rect.y + rect.h < p.y || rect.y > p.y + p.h))) continue;
        place = rect;
        break;
      }
      if (place) {
        placed.push(place);
        ctx.fillStyle = 'rgba(8,8,14,0.72)';
        ctx.fillRect(place.x, place.y, place.w, place.h);
        ctx.fillStyle = color;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, place.x + 2, place.y + 7.4);
      }
      ctx.restore();
    };
    const placed = [];
    marker(res.stats.base, 'diamond', 'rgba(159,212,255,0.95)', '基线');
    marker(res.stats.mean, 'cross', 'rgba(255,233,168,0.95)', '均值');
    marker(res.stats.mode, 'plus', '#ffffff', '众数');

    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

    drawInset(res);
  }

  /* 右上角全景小图：整个门框 + 当前视野 + 三个标记 */
  function insetRect() {
    const W = canvasCss.width;
    const iw = Math.min(200, Math.max(118, W * 0.30));
    return { x: W - iw - 12, y: 12, w: iw, h: iw / 2.36 };
  }

  function drawInset(res) {
    const r = insetRect();
    const iy0 = -4.25, iy1 = 4.25, iz0 = -0.55, iz1 = 3.05;
    const mx = (y) => ((y - iy0) / (iy1 - iy0)) * r.w;
    const my = (z) => r.h - ((z - iz0) / (iz1 - iz0)) * r.h;
    const path = new Path2D();
    if (path.roundRect) path.roundRect(r.x, r.y, r.w, r.h, 7);
    else path.rect(r.x, r.y, r.w, r.h);

    ctx.save();
    ctx.fillStyle = 'rgba(7,6,13,0.88)';
    ctx.fill(path);
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.lineWidth = 1;
    ctx.stroke(path);
    ctx.clip(path);
    ctx.translate(r.x, r.y);

    /* 门框 */
    const gx0 = mx(-C.halfWidth), gx1 = mx(C.halfWidth);
    const gy0 = my(C.goalHeight), gy1 = my(0);
    ctx.fillStyle = 'rgba(126,226,168,0.12)';
    ctx.fillRect(gx0, gy0, gx1 - gx0, gy1 - gy0);
    ctx.strokeStyle = 'rgba(126,226,168,0.85)';
    ctx.lineWidth = 1;
    ctx.strokeRect(gx0 + 0.5, gy0 + 0.5, gx1 - gx0 - 1, gy1 - gy0 - 1);

    /* 当前视野 */
    const { win } = res;
    const vx0 = Math.max(0, Math.min(r.w, mx(win.yl)));
    const vx1 = Math.max(0, Math.min(r.w, mx(win.yh)));
    const vy0 = Math.max(0, Math.min(r.h, my(win.zh)));
    const vy1 = Math.max(0, Math.min(r.h, my(win.zl)));
    if (vx1 - vx0 > 1 && vy1 - vy0 > 1) {
      ctx.fillStyle = 'rgba(201,182,255,0.16)';
      ctx.fillRect(vx0, vy0, vx1 - vx0, vy1 - vy0);
      ctx.strokeStyle = 'rgba(201,182,255,0.95)';
      ctx.lineWidth = 1.2;
      ctx.strokeRect(vx0 + 0.6, vy0 + 0.6, Math.max(1, vx1 - vx0 - 1.2), Math.max(1, vy1 - vy0 - 1.2));
    }

    /* 标记 */
    const dot = (pt, color, kind) => {
      if (!pt) return;
      const x = mx(pt[0]), y = my(pt[1]);
      if (x < -3 || x > r.w + 3 || y < -3 || y > r.h + 3) return;
      ctx.strokeStyle = 'rgba(8,8,14,0.6)';
      ctx.fillStyle = color;
      ctx.lineWidth = 2.6;
      if (kind === 'dot') {
        ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2);
        ctx.strokeStyle = color; ctx.fill();
      } else {
        ctx.beginPath();
        if (kind === 'cross') {
          ctx.moveTo(x - 3.4, y); ctx.lineTo(x + 3.4, y);
          ctx.moveTo(x, y - 3.4); ctx.lineTo(x, y + 3.4);
        } else {
          ctx.moveTo(x, y - 3.4); ctx.lineTo(x + 3.4, y); ctx.lineTo(x, y + 3.4); ctx.lineTo(x - 3.4, y);
          ctx.closePath();
        }
        ctx.stroke();
      }
    };
    dot(res.stats.base, 'rgba(159,212,255,0.95)', 'diamond');
    dot(res.stats.mean, 'rgba(255,233,168,0.95)', 'cross');
    dot(res.stats.mode, '#ffffff', 'dot');

    ctx.fillStyle = 'rgba(200,195,220,0.45)';
    ctx.font = '9px system-ui,"Microsoft YaHei",sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('门框全景', 6, r.h - 4);
    ctx.restore();
  }

  /* ───────────────────────────── 读数 ───────────────────────────── */
  function updateReadout(res) {
    const { mode, mean, pGoal, a50, a95, base: basePt } = res.stats;
    $('#s-mode').textContent = mode ? `(${fmt(mode[0])}, ${fmt(mode[1])}) m` : '—';
    $('#s-mean').textContent = `(${fmt(mean[0])}, ${fmt(mean[1])}) m`;
    const goalEl = $('#s-goal');
    goalEl.innerHTML = `${(pGoal * 100).toFixed(1)}%<small>空中过门条件下</small>`;
    goalEl.className = pGoal >= 0.5 ? 'is-goal' : 'is-miss';
    $('#s-a50').textContent = `${a50.toFixed(3)} m²`;
    $('#s-a95').textContent = `${a95.toFixed(3)} m²`;
    $('#s-base').textContent = basePt ? `(${fmt(basePt[0])}, ${fmt(basePt[1])}) m` : '无法求解';

    const note = $('#fig-note');
    if (mode) {
      const [my2, mz2] = mode;
      const post = C.halfWidth - Math.abs(my2);
      const bar = C.goalHeight - mz2;
      const parts = [
        post >= 0 ? `距${my2 >= 0 ? '右' : '左'}门柱 ${post.toFixed(2)} m` : `越过${my2 >= 0 ? '右' : '左'}门柱 ${(-post).toFixed(2)} m`,
        bar >= 0 ? `横梁下方 ${bar.toFixed(2)} m` : `高出横梁 ${(-bar).toFixed(2)} m`,
      ].filter(Boolean);
      note.textContent = `众数位置：${parts.join(' · ')}。白线为 50% 区域、青线为 95% 区域；右上小图是门框全景，紫框即当前视野。`;
    } else {
      note.textContent = '白线为 50% 区域、青线为 95% 区域；右上小图是门框全景，紫框即当前视野。';
    }

    canvas.setAttribute('aria-label',
      `球门平面落点：众数 (${fmt(mode ? mode[0] : NaN)}, ${fmt(mode ? mode[1] : NaN)}) 米，`
      + `分布均值 (${fmt(mean[0])}, ${fmt(mean[1])}) 米，`
      + `入门概率 ${(pGoal * 100).toFixed(1)}%，50% 区域 ${a50.toFixed(2)} 平方米，95% 区域 ${a95.toFixed(2)} 平方米。`);

    const rows = FEATURE_INFO.map(([label, unit], i) => {
      const raw = res.feats.display[i];
      const digits = i >= 9 ? 0 : 2;
      const value = Number.isFinite(raw) ? raw.toFixed(digits) + ' ' + unit : '无解';
      const z = res.zscores ? Math.abs(res.zscores[i]) : 0;
      return `<tr><td>${label}</td><td>${value}</td><td class="${z > 3 ? 'big' : ''}">${i < 10 ? z.toFixed(2) : '—'}</td></tr>`;
    }).join('');
    $('#feature-rows').innerHTML = rows;
  }

  function warnFor(res) {
    const warn = $('#warn');
    const msgs = [];
    if (!Number.isFinite(res.feats.tGoal)) {
      msgs.push('这三个点推不出正向的过门时间（x 方向速度不足或无解）。网络仍会给出答案，但那已远在训练分布之外。');
    } else if (res.maxZ > 3) {
      msgs.push(`输入偏离训练分布较远（标准化偏差最大 ${res.maxZ.toFixed(1)}σ，参考阈值 3），结果只能当作外推参考。`);
    }
    warn.textContent = msgs.join(' ');
    warn.hidden = !msgs.length;
  }

  /* ───────────────────────────── 交互 ───────────────────────────── */
  function readInputs() {
    let ok = true;
    for (const el of $$('#panel input[type="number"]')) {
      const p = Number(el.dataset.point);
      const a = Number(el.dataset.axis);
      const v = el.value.trim() === '' ? NaN : Number(el.value);
      if (!Number.isFinite(v)) { el.classList.add('is-bad'); ok = false; continue; }
      el.classList.remove('is-bad');
      state.points[p][a] = v;
    }
    return ok;
  }

  let pending = null;
  let lastAspect = 4 / 3;

  function scheduleCompute(delay = 80) {
    if (pending) return;
    pending = setTimeout(() => { pending = null; requestAnimationFrame(run); }, delay);
  }

  function run() {
    if (!state.meta) return;
    if (!readInputs()) return;
    let res;
    const t0 = performance.now();
    try {
      res = compute(lastAspect);
    } catch (err) {
      console.error(err);
      return;
    }
    state.last = res;
    renderBase(res);
    paint();
    updateReadout(res);
    warnFor(res);
    syncUrl();
    if (location.hash === '#debug') {
      window.__pen = {
        win: res.win, nx: res.field.nx, ny: res.field.ny, aspect: lastAspect,
        comps: res.comps, stats: res.stats, maxZ: res.maxZ, ms: performance.now() - t0,
      };
    }
  }

  function syncUrl() {
    const enc = (arr) => arr.map((v) => Math.round(v * 1000) / 1000).join(',');
    const params = new URLSearchParams();
    params.set('p1', enc(state.points[1]));
    params.set('p2', enc(state.points[2]));
    params.set('sigma', String(state.sigma));
    params.set('seed', state.seed);
    try {
      history.replaceState(null, '', location.pathname + '?' + params.toString() + location.hash);
    } catch (e) { /* file:// 下忽略 */ }
  }

  function readUrl() {
    const params = new URLSearchParams(location.search);
    const parse = (key) => {
      const raw = params.get(key);
      if (!raw) return null;
      const vals = raw.split(',').map(Number);
      return vals.length === 3 && vals.every(Number.isFinite) ? vals : null;
    };
    const p1 = parse('p1');
    const p2 = parse('p2');
    if (p1) state.points[1] = p1;
    if (p2) state.points[2] = p2;
    const sig = Number(params.get('sigma'));
    if ([2, 5, 10, 15].includes(sig)) state.sigma = sig;
    const seed = params.get('seed');
    if (['11', '22', '33', 'avg'].includes(seed)) state.seed = seed;
  }

  function applyStateToInputs() {
    for (const el of $$('#panel input[type="number"]')) {
      const p = Number(el.dataset.point);
      const a = Number(el.dataset.axis);
      el.value = fmtInput(state.points[p][a]);
      el.classList.remove('is-bad');
    }
    for (const el of $$('input[name="sigma"]')) el.checked = Number(el.value) === state.sigma;
    for (const el of $$('input[name="seed"]')) el.checked = el.value === state.seed;
  }

  /* 预设来自同一个物理模拟器（seed 20261108）抽样，过门点分别为
     左上角、右上角、右下角、中路、门柱外 */
  const PRESETS = [
    { id: 'case', name: '题面案例', points: CASE_POINTS },
    { id: 'tl', name: '左上死角', points: [[0, 0, 0.11], [0.955, -0.325, 0.469], [1.903, -0.63, 0.79]] },
    { id: 'tr', name: '右上死角', points: [[0, 0, 0.11], [1.388, 0.475, 0.518], [2.773, 0.917, 0.879]] },
    { id: 'br', name: '右下死角', points: [[0, 0, 0.11], [1.534, 0.466, 0.303], [3.047, 0.895, 0.447]] },
    { id: 'mid', name: '慢速中路', points: [[0, 0, 0.11], [0.965, 0.017, 0.394], [1.916, 0.035, 0.655]] },
    { id: 'wide', name: '偏出门柱', points: [[0, 0, 0.11], [1.382, 0.514, 0.39], [2.735, 1.034, 0.64]] },
  ];

  function buildPresets() {
    const wrap = $('#presets');
    wrap.innerHTML = '';
    for (const preset of PRESETS) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = preset.name;
      b.dataset.preset = preset.id;
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => {
        state.points = preset.points.map((p) => p.slice());
        applyStateToInputs();
        $$('#presets button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
        b.setAttribute('aria-pressed', 'true');
        run();
      });
      wrap.appendChild(b);
    }
  }

  function wireEvents() {
    for (const el of $$('#panel input[type="number"]')) {
      el.addEventListener('input', () => {
        $$('#presets button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
        scheduleCompute(70);
      });
      el.addEventListener('blur', () => {
        const v = Number(el.value);
        if (Number.isFinite(v)) el.value = fmtInput(v);
      });
    }
    for (const el of $$('input[name="sigma"]')) {
      el.addEventListener('change', () => {
        if (!el.checked) return;
        state.sigma = Number(el.value);
        run();
      });
    }
    for (const el of $$('input[name="seed"]')) {
      el.addEventListener('change', () => {
        if (!el.checked) return;
        state.seed = el.value;
        run();
      });
    }
    $('#reset').addEventListener('click', () => {
      state.points = CASE_POINTS.map((p) => p.slice());
      state.sigma = 5;
      state.seed = '11';
      applyStateToInputs();
      $$('#presets button').forEach((x, i) => x.setAttribute('aria-pressed', i === 0 ? 'true' : 'false'));
      run();
    });
    $('#copy-link').addEventListener('click', async (ev) => {
      const btn = ev.currentTarget;
      try {
        await navigator.clipboard.writeText(location.href);
        btn.textContent = '已复制';
        btn.classList.add('is-done');
      } catch (err) {
        btn.textContent = '复制失败';
      }
      setTimeout(() => {
        btn.textContent = '复制当前链接';
        btn.classList.remove('is-done');
      }, 1600);
    });

    const moveTo = (clientX, clientY) => {
      const res = state.last;
      if (!res) return;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      const ins = insetRect();
      if (x > ins.x - 4 && x < ins.x + ins.w + 4 && y > ins.y - 4 && y < ins.y + ins.h + 4) {
        $('#tip').hidden = true;
        if (state.hover) { state.hover = null; paint(); }
        return;
      }
      const { win } = res;
      const wy = win.yl + (x / rect.width) * (win.yh - win.yl);
      const wz = win.zl + (1 - y / rect.height) * (win.zh - win.zl);
      state.hover = { y: wy, z: wz };
      const d = densityAt(res.dens, wy, wz);
      const inside = Math.abs(wy) < C.halfWidth && wz > 0 && wz < C.goalHeight;
      let band = '概率区外';
      if (d >= res.level50) band = '50% 区内';
      else if (d >= res.level95) band = '95% 区内';
      const tip = $('#tip');
      tip.hidden = false;
      tip.textContent = `Y ${wy.toFixed(2)} · Z ${wz.toFixed(2)} · ${band}${inside ? ' · 门内' : ' · 门外'}`;
      const px = Math.max(60, Math.min(rect.width - 60, x));
      tip.style.left = px + 'px';
      tip.style.top = Math.max(30, y) + 'px';
      paint();
    };
    canvas.addEventListener('mousemove', (ev) => moveTo(ev.clientX, ev.clientY));
    canvas.addEventListener('mouseleave', () => {
      state.hover = null;
      $('#tip').hidden = true;
      paint();
    });
    canvas.addEventListener('touchstart', (ev) => {
      if (ev.touches.length === 1) moveTo(ev.touches[0].clientX, ev.touches[0].clientY);
    }, { passive: true });
    canvas.addEventListener('touchend', () => {
      state.hover = null;
      $('#tip').hidden = true;
      paint();
    });

    let timer = null;
    window.addEventListener('resize', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        resizeCanvas();
        lastAspect = canvasCss.width / canvasCss.height;
        if (state.meta) run();
      }, 140);
    });
  }

  /* ───────────────────────────── 启动 ───────────────────────────── */
  async function main() {
    readUrl();
    applyStateToInputs();
    buildPresets();
    wireEvents();
    resizeCanvas();
    lastAspect = canvasCss.width / canvasCss.height;
    try {
      await loadModel();
    } catch (err) {
      console.error(err);
      $('#loading').textContent = '模型载入失败：' + err.message;
      return;
    }
    $('#loading').hidden = true;
    $('#model-badge').textContent = `P1 · ${state.meta.arch.parameters.toLocaleString('en-US')} 参数`;
    const first = $('#presets button');
    if (first) first.setAttribute('aria-pressed', 'true');
    run();
  }

  main();
})();
