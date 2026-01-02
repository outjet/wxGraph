const PANEL_CONFIG = [
  { key: 'temp', elementId: 'temp-panel', title: 'Temperature', yTitle: '°F' },
  { key: 'cloud', elementId: 'cloud-panel', title: 'Cloud Cover', yTitle: '%', yRange: [0, 100] },
  { key: 'qpf', elementId: 'qpf-panel', title: 'QPF', yTitle: 'inches' },
  { key: 'snow', elementId: 'snow-panel', title: 'Snow', yTitle: 'inches' },
  { key: 'ice', elementId: 'ice-panel', title: 'Freezing Rain + Sleet', yTitle: 'inches', barmode: 'stack' },
  { key: 'wind', elementId: 'wind-panel', title: 'Wind', yTitle: 'mph' },
];

const MODEL_COLORS = {
  HRRR: '#001219',
  GFS: '#005F73',
  RAP: '#0A9396',
  GEFS: '#94D2BD',
  NAM: '#E9D8A6',
  AIGFS: '#CA6702',
  HGEFS: '#9B2226',
};

const hexToRgb = (hex) => {
  const cleaned = hex.replace('#', '');
  const num = parseInt(cleaned, 16);
  return {
    r: (num >> 16) & 255,
    g: (num >> 8) & 255,
    b: num & 255,
  };
};

const rgbToHex = ({ r, g, b }) =>
  `#${[r, g, b].map((value) => value.toString(16).padStart(2, '0')).join('')}`;

const rgbToHsl = ({ r, g, b }) => {
  const rNorm = r / 255;
  const gNorm = g / 255;
  const bNorm = b / 255;
  const max = Math.max(rNorm, gNorm, bNorm);
  const min = Math.min(rNorm, gNorm, bNorm);
  const delta = max - min;
  let h = 0;
  let s = 0;
  const l = (max + min) / 2;

  if (delta !== 0) {
    s = delta / (1 - Math.abs(2 * l - 1));
    switch (max) {
      case rNorm:
        h = ((gNorm - bNorm) / delta) % 6;
        break;
      case gNorm:
        h = (bNorm - rNorm) / delta + 2;
        break;
      default:
        h = (rNorm - gNorm) / delta + 4;
    }
    h *= 60;
    if (h < 0) h += 360;
  }

  return { h, s, l };
};

const hslToRgb = ({ h, s, l }) => {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;
  if (h >= 0 && h < 60) {
    r = c;
    g = x;
  } else if (h < 120) {
    r = x;
    g = c;
  } else if (h < 180) {
    g = c;
    b = x;
  } else if (h < 240) {
    g = x;
    b = c;
  } else if (h < 300) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }
  return {
    r: Math.round((r + m) * 255),
    g: Math.round((g + m) * 255),
    b: Math.round((b + m) * 255),
  };
};

const lightenHex = (hex, deltaL = 0.18) => {
  const hsl = rgbToHsl(hexToRgb(hex));
  const l = Math.min(0.95, hsl.l + deltaL);
  return rgbToHex(hslToRgb({ ...hsl, l }));
};

const hexToRgba = (hex, alpha) => {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const getModelColor = (model) => MODEL_COLORS[model] || '#0A9396';

async function loadData() {
  const response = await fetch('/meteogram_latest.json', { cache: 'no-cache' });
  if (!response.ok) {
    throw new Error('Failed to load meteogram data');
  }
  return response.json();
}

function extractModels(record) {
  const models = new Set();
  Object.keys(record).forEach((key) => {
    if (key === 'valid_time') return;
    const [model] = key.split('_');
    if (model) {
      models.add(model);
    }
  });
  return Array.from(models).sort();
}

function buildControls(models, onToggle) {
  const controls = document.getElementById('controls');
  controls.innerHTML = '';
  const title = document.createElement('strong');
  title.textContent = 'Models:';
  controls.appendChild(title);

  models.forEach((model) => {
    const label = document.createElement('label');
    label.className = 'model-toggle';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = true;
    checkbox.dataset.model = model;
    checkbox.addEventListener('change', () => onToggle(model, checkbox.checked));
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(model));
    controls.appendChild(label);
  });
}

function collectTraces(data, models) {
  const traces = [];
  const xAll = data.map((row) => row.valid_time);
  const getField = (model, suffix) => `${model}_${suffix}`;
  const buildSeries = (column) => {
    const xs = [];
    const ys = [];

    for (const row of data) {
      const raw = row[column];
      if (raw === undefined || raw === null || raw === '') continue;
      const num = Number(raw);
      if (!Number.isFinite(num)) continue;
      xs.push(row.valid_time);
      ys.push(num);
    }

    return { xs, ys };
  };
  const columnExistsAnywhere = (column) =>
    data.some((row) => row[column] !== undefined && row[column] !== null && row[column] !== '');
  const localFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZoneName: 'short',
  });

  const resolveField = (model, field, fallbacks = []) => {
    const column = getField(model, field);
    if (columnExistsAnywhere(column)) {
      return { column, label: field };
    }
    for (const fallback of fallbacks) {
      const fallbackColumn = getField(model, fallback);
      if (columnExistsAnywhere(fallbackColumn)) {
        return { column: fallbackColumn, label: fallback };
      }
    }
    return null;
  };

  const addTrace = (model, field, panel, options = {}, fallbacks = []) => {
    const resolved = resolveField(model, field, fallbacks);
    if (!resolved) return;
    const { column, label } = resolved;
    const { xs, ys } = buildSeries(column);
    if (xs.length === 0) return;
    const type = options.type || 'scatter';
    const sparsity = xs.length / Math.max(1, data.length);
    const defaultMode = sparsity < 0.75 ? 'lines+markers' : 'lines';
    const baseColor = options.color || getModelColor(model);
    const lineColor = options.lineColor || baseColor;
    const trace = {
      x: xs,
      y: ys,
      name: `${model} ${label.replace(/_/g, ' ')}`,
      type,
      meta: { model, panel },
      text: xs.map((value) => localFormatter.format(new Date(value))),
      hovertemplate: `${model}: %{y:.2f}<br>%{text}<extra></extra>`,
    };
    if (type === 'bar') {
      const marker = options.marker || { opacity: 0.7 };
      trace.marker = { color: baseColor, ...marker };
    } else {
      trace.mode = options.mode || defaultMode;
      trace.connectgaps = options.connectgaps ?? false;
      trace.line = {
        dash: options.dash || 'solid',
        shape: options.shape || 'linear',
        color: lineColor,
      };
      if (trace.mode.includes('markers')) {
        const marker = options.marker || { size: 6 };
        trace.marker = { color: lineColor, ...marker };
      }
    }
    if (options.fill) {
      trace.fill = options.fill;
      trace.fillcolor = options.fillcolor || hexToRgba(baseColor, 0.25);
    }
    traces.push(trace);
  };

  const buildSeriesMap = (column) => {
    const map = new Map();
    for (const row of data) {
      const raw = row[column];
      if (raw === undefined || raw === null || raw === '') continue;
      const num = Number(raw);
      if (!Number.isFinite(num)) continue;
      map.set(row.valid_time, num);
    }
    return map;
  };

  const addIceComposite = (model) => {
    const iceFields = ['cip_in', 'bfp_in'];
    const resolved = iceFields
      .map((field) => resolveField(model, field))
      .filter(Boolean);
    if (!resolved.length) return;

    const baseColor = getModelColor(model);
    resolved.forEach((entry, index) => {
      const color = index === 0 ? baseColor : lightenHex(baseColor, 0.12);
      addTrace(
        model,
        entry.label,
        'ice',
        { type: 'bar', marker: { opacity: 0.6, color } },
      );
    });

    const seriesMaps = resolved.map((entry) => buildSeriesMap(entry.column));
    const times = Array.from(new Set(seriesMaps.flatMap((map) => Array.from(map.keys())))).sort();
    if (!times.length) return;
    const xs = [];
    const ys = [];
    let running = 0;
    times.forEach((time) => {
      let step = 0;
      seriesMaps.forEach((map) => {
        step += map.get(time) || 0;
      });
      running += step;
      xs.push(time);
      ys.push(running);
    });
    traces.push({
      x: xs,
      y: ys,
      name: `${model} ice accum`,
      type: 'scatter',
      mode: 'lines',
      line: { dash: 'dot', color: baseColor },
      meta: { model, panel: 'ice' },
      text: xs.map((value) => localFormatter.format(new Date(value))),
      hovertemplate: `${model}: %{y:.2f}<br>%{text}<extra></extra>`,
    });
  };

  models.forEach((model) => {
    addTrace(model, 'temp_f', 'temp');
    const tempLight = lightenHex(getModelColor(model), 0.18);
    addTrace(model, 'apparent_temp_f', 'temp', { dash: 'dash', lineColor: tempLight });
    addTrace(model, 'cloud_pct', 'cloud', { fill: 'tozeroy' });
    addTrace(
      model,
      'qpf_in',
      'qpf',
      { type: 'bar', marker: { opacity: 0.7 } },
      ['ipf_in'],
    );
    addTrace(model, 'qpf_in_accum', 'qpf', { dash: 'dot' }, ['qpf_in']);
    addTrace(
      model,
      'snowfall_in',
      'snow',
      { type: 'bar', marker: { opacity: 0.7 } },
      ['snowsfc_in', 'snow_in_accum'],
    );
    addTrace(model, 'snow_acc_in', 'snow', {}, ['snow_in_accum']);
    addIceComposite(model);
    addTrace(model, 'wind10m_mph', 'wind', {}, ['gust_mph']);
  });

  traces.push({
    x: xAll,
    y: xAll.map(() => 32),
    name: '32°F',
    type: 'scatter',
    mode: 'lines',
    line: { dash: 'dot', color: '#777' },
    hoverinfo: 'skip',
    meta: { model: 'FREEZING', panel: 'temp' },
  });

  return traces;
}

function filterTraces(traces, visibility) {
  return traces.filter((trace) => visibility[trace.meta.model] !== false || trace.meta.model === 'FREEZING');
}

function baseLayout(title, yTitle, options = {}) {
  const yaxis = { title: yTitle };
  if (options.yRange) {
    yaxis.range = options.yRange;
  }
  return {
    title,
    height: 420,
    yaxis,
    xaxis: { type: 'date' },
    margin: { t: 40, r: 20, b: 40, l: 55 },
    legend: { orientation: 'h' },
    hovermode: 'x unified',
    barmode: options.barmode,
  };
}

function drawPanels(traces, visibility) {
  const filtered = filterTraces(traces, visibility);
  const plots = {};

  PANEL_CONFIG.forEach(({ key, elementId, title, yTitle, yRange, barmode }) => {
    const panelTraces = filtered.filter((trace) => trace.meta.panel === key);
    const element = document.getElementById(elementId);
    console.log(`[panel=${key}] elementId=${elementId} element=`, element, `traces=${panelTraces.length}`);
    if (panelTraces.length === 0) {
      Plotly.react(
        element,
        [],
        {
          ...baseLayout(title, yTitle, { yRange, barmode }),
          annotations: [
            {
              text: 'No data columns found for this panel',
              xref: 'paper',
              yref: 'paper',
              x: 0.5,
              y: 0.5,
              showarrow: false,
              font: { size: 14 },
            },
          ],
        },
        { responsive: true },
      );
      return;
    }
    Plotly.react(element, panelTraces, baseLayout(title, yTitle, { yRange, barmode }), { responsive: true });
    plots[key] = element;
  });

  syncXAxes(plots);
}

function syncXAxes(plots) {
  let updating = false;
  const keys = Object.keys(plots);
  keys.forEach((key) => {
    plots[key].on('plotly_relayout', (eventData) => {
      if (updating) return;
      if (!eventData || !('xaxis.range[0]' in eventData)) return;
      updating = true;
      const range = [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']];
      keys.forEach((otherKey) => {
        if (otherKey === key) return;
        Plotly.relayout(plots[otherKey], 'xaxis.range', range);
      });
      updating = false;
    });
  });
}

async function init() {
  try {
    const data = await loadData();
    if (!data.length) {
      throw new Error('No meteogram data available');
    }
    const models = extractModels(data[0]);
    if (!models.length) {
      throw new Error('No model columns detected in meteogram JSON.');
    }

    const traces = collectTraces(data, models);
    const visibility = Object.fromEntries(models.map((model) => [model, true]));

    const update = () => drawPanels(traces, visibility);
    buildControls(models, (model, visible) => {
      visibility[model] = visible;
      update();
    });

    update();
  } catch (error) {
    const controls = document.getElementById('controls');
    controls.textContent = error.message;
  }
}

init();
