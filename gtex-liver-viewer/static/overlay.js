// The WebGL point layer over an OpenSeadragon viewer, adapted from cosine_viewer's WsiGLOverlay: every cell of
// a slide in one draw call, at its level-0 px position; colour and visibility come from a table indexed by the
// cell's code (HistoPLUS type or cluster label), so switching layers or hiding a code only changes a uniform.
// A code can be drawn in front (alpha 1) or behind (alpha 2: a first pass under everything else), and each
// code has its own size factor, so a few highlighted cells stay visible over the rest. A density threshold hides
// the cells below it in every layer: decided by the cell's density bin where the bins differ (exact: the server
// bins in float64) and by its float32 density inside the threshold's own bin.

const MAX_TYPES = 16;

const VS = `
  attribute vec2 a_position;    // level-0 px
  attribute float a_type;       // HistoPLUS type code
  attribute float a_bin;        // density bin (cohort decile)
  attribute float a_density;    // 10-NN density, cells per mm²
  uniform float u_minBin;       // the bin the density threshold falls in
  uniform float u_minDensity;   // the density threshold
  uniform vec4 u_bounds;        // visible image rectangle (x, y, width, height) in level-0 px
  uniform float u_pointSize;    // device px
  uniform vec4 u_colors[${MAX_TYPES}];  // rgb, and 0 hidden, 1 in front, 2 behind
  uniform float u_sizes[${MAX_TYPES}];  // size factor per code
  uniform float u_pass;         // 2 draws the codes behind, 1 the codes in front
  varying vec3 v_color;
  void main() {
    int code = int(a_type + 0.5);
    vec4 colour = u_colors[code];
    bool below = a_bin < u_minBin - 0.5 || (a_bin < u_minBin + 0.5 && a_density < u_minDensity);
    if (below || colour.a < 0.5 || abs(colour.a - u_pass) > 0.5) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      gl_PointSize = 0.0;
      return;
    }
    vec2 p = (a_position - u_bounds.xy) / u_bounds.zw;
    gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    gl_PointSize = u_pointSize * u_sizes[code];
    v_color = colour.rgb;
  }`;

const FS = `
  precision mediump float;
  uniform float u_alpha;
  varying vec3 v_color;
  void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    gl_FragColor = vec4(v_color, u_alpha);
  }`;

export class PointsOverlay {
  constructor(viewer, cellPx) {
    this.viewer = viewer;
    this.cellPx = cellPx; // a cell's diameter in level-0 px, for the point size when zoomed in
    this.canvas = document.createElement("canvas");
    Object.assign(this.canvas.style, { position: "absolute", top: "0", left: "0", pointerEvents: "none" });
    viewer.canvas.appendChild(this.canvas);
    const gl = this.canvas.getContext("webgl", { antialias: false });
    if (!gl) throw new Error("WebGL is not available in this browser");
    this.gl = gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    this.program = this.createProgram(VS, FS);
    this.loc = {
      position: gl.getAttribLocation(this.program, "a_position"),
      type: gl.getAttribLocation(this.program, "a_type"),
      bin: gl.getAttribLocation(this.program, "a_bin"),
      density: gl.getAttribLocation(this.program, "a_density"),
      minBin: gl.getUniformLocation(this.program, "u_minBin"),
      minDensity: gl.getUniformLocation(this.program, "u_minDensity"),
      bounds: gl.getUniformLocation(this.program, "u_bounds"),
      pointSize: gl.getUniformLocation(this.program, "u_pointSize"),
      colors: gl.getUniformLocation(this.program, "u_colors"),
      sizes: gl.getUniformLocation(this.program, "u_sizes"),
      pass: gl.getUniformLocation(this.program, "u_pass"),
      alpha: gl.getUniformLocation(this.program, "u_alpha"),
    };
    this.positions = gl.createBuffer();
    this.types = gl.createBuffer();
    this.bins = gl.createBuffer();
    this.densities = gl.createBuffer();
    this.count = 0;
    this.minBin = 0;
    this.minDensity = 0;
    this.colors = new Float32Array(MAX_TYPES * 4);
    this.sizes = new Float32Array(MAX_TYPES).fill(1);
    this.pointScale = 1;
    this.alpha = 0.85;
    this._draw = () => this.draw();
    this._resize = () => this.resize();
    viewer.addHandler("update-viewport", this._draw);
    viewer.addHandler("resize", this._resize);
    this.resize();
  }

  createProgram(vsSource, fsSource) {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vsSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fsSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    const { clientWidth, clientHeight } = this.viewer.container;
    this.canvas.width = clientWidth * ratio;
    this.canvas.height = clientHeight * ratio;
    this.canvas.style.width = `${clientWidth}px`;
    this.canvas.style.height = `${clientHeight}px`;
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.draw();
  }

  // xy: Float32Array of 2n level-0 px coordinates; types: Uint8Array of n codes; bins: Uint8Array of n density
  // bins; densities: Float32Array of n densities.
  setPoints(xy, types, bins, densities) {
    const gl = this.gl;
    const n = types.length;
    if (xy.length !== 2 * n || bins.length !== n || densities.length !== n) {
      throw new Error(`${xy.length / 2} positions, ${n} codes, ${bins.length} bins, ${densities.length} densities`);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positions);
    gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.types);
    gl.bufferData(gl.ARRAY_BUFFER, types, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bins);
    gl.bufferData(gl.ARRAY_BUFFER, bins, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.densities);
    gl.bufferData(gl.ARRAY_BUFFER, densities, gl.STATIC_DRAW);
    this.count = n;
    this.draw();
  }

  // Hide the cells below a density threshold: those in a lower bin, and those in its bin below its value.
  setThreshold(minBin, minDensity) { this.minBin = minBin; this.minDensity = minDensity; this.draw(); }

  // Replace the per-cell codes the colour table is indexed by (type codes, or cluster labels), one per cell.
  setCodes(codes) {
    const gl = this.gl;
    if (codes.length !== this.count) throw new Error(`${codes.length} codes for ${this.count} cells`);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.types);
    gl.bufferData(gl.ARRAY_BUFFER, codes, gl.STATIC_DRAW);
    this.draw();
  }

  // colors: Float32Array of MAX_TYPES × (r, g, b, state), indexed by code; state 0 hidden, 1 front, 2 behind.
  // sizes: optional Float32Array of MAX_TYPES size factors (1 when left out).
  setColors(colors, sizes) {
    this.colors.fill(0);
    this.colors.set(colors.subarray(0, MAX_TYPES * 4));
    this.sizes.fill(1);
    if (sizes) this.sizes.set(sizes.subarray(0, MAX_TYPES));
    this.draw();
  }

  setPointScale(scale) { this.pointScale = scale; this.draw(); }

  setAlpha(alpha) { this.alpha = alpha; this.draw(); }

  draw() {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const item = this.viewer.world.getItemAt(0);
    if (!this.count || !item) return;
    const rect = item.viewportToImageRectangle(this.viewer.viewport.getBounds(true));
    const ratio = window.devicePixelRatio || 1;
    const cssPerPx = this.canvas.width / ratio / rect.width;
    const size = Math.min(48, Math.max(1.5, this.cellPx * cssPerPx)) * this.pointScale * ratio;
    gl.useProgram(this.program);
    gl.uniform4f(this.loc.bounds, rect.x, rect.y, rect.width, rect.height);
    gl.uniform1f(this.loc.pointSize, size);
    gl.uniform1f(this.loc.alpha, this.alpha);
    gl.uniform4fv(this.loc.colors, this.colors);
    gl.uniform1fv(this.loc.sizes, this.sizes);
    gl.uniform1f(this.loc.minBin, this.minBin);
    gl.uniform1f(this.loc.minDensity, this.minDensity);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positions);
    gl.enableVertexAttribArray(this.loc.position);
    gl.vertexAttribPointer(this.loc.position, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.types);
    gl.enableVertexAttribArray(this.loc.type);
    gl.vertexAttribPointer(this.loc.type, 1, gl.UNSIGNED_BYTE, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bins);
    gl.enableVertexAttribArray(this.loc.bin);
    gl.vertexAttribPointer(this.loc.bin, 1, gl.UNSIGNED_BYTE, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.densities);
    gl.enableVertexAttribArray(this.loc.density);
    gl.vertexAttribPointer(this.loc.density, 1, gl.FLOAT, false, 0, 0);
    let behind = false;
    for (let i = 3; i < this.colors.length; i += 4) behind ||= this.colors[i] > 1.5;
    if (behind) {
      gl.uniform1f(this.loc.pass, 2);
      gl.drawArrays(gl.POINTS, 0, this.count);
    }
    gl.uniform1f(this.loc.pass, 1);
    gl.drawArrays(gl.POINTS, 0, this.count);
  }
}
