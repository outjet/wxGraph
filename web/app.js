const PANEL_CONFIG = [
  { key: 'temp', elementId: 'temp-panel', title: 'Temperature', yTitle: '°F' },
  { key: 'qpf', elementId: 'qpf-panel', title: 'QPF', yTitle: 'inches' },
  { key: 'snow', elementId: 'snow-panel', title: 'Snow', yTitle: 'inches' },
  { key: 'wind', elementId: 'wind-panel', title: 'Wind', yTitle: 'mph' },
];

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

  const addTrace = (model, field, panel, options = {}) => {
    const column = getField(model, field);
    if (!columnExistsAnywhere(column)) return;
    const { xs, ys } = buildSeries(column);
    if (xs.length === 0) return;
    const type = options.type || 'scatter';
    const sparsity = xs.length / Math.max(1, data.length);
    const defaultMode = sparsity < 0.75 ? 'lines+markers' : 'lines';
    const trace = {
      x: xs,
      y: ys,
      name: `${model} ${field.replace(/_/g, ' ')}`,
      type,
      meta: { model, panel },
      hovertemplate: `${model}: %{y:.2f}<br>%{x|%b %d %H:%MZ}<extra></extra>`,
    };
    if (type === 'bar') {
      trace.marker = options.marker || { opacity: 0.7 };
    } else {
      trace.mode = options.mode || defaultMode;
      trace.connectgaps = options.connectgaps ?? false;
      trace.line = {
        dash: options.dash || 'solid',
        shape: options.shape || 'linear',
      };
      if (trace.mode.includes('markers')) {
        trace.marker = options.marker || { size: 6 };
      }
    }
    traces.push(trace);
  };

  models.forEach((model) => {
    addTrace(model, 'temp_f', 'temp');
    addTrace(model, 'apparent_temp_f', 'temp', { dash: 'dash' });
    addTrace(model, 'qpf_in', 'qpf', { type: 'bar', marker: { opacity: 0.7 } });
    addTrace(model, 'snowfall_in', 'snow', { type: 'bar', marker: { opacity: 0.7 } });
    addTrace(model, 'snow_acc_in', 'snow');
    addTrace(model, 'wind10m_mph', 'wind');
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

function baseLayout(title, yTitle) {
  return {
    title,
    height: 420,
    yaxis: { title: yTitle },
    xaxis: { type: 'date' },
    margin: { t: 40, r: 20, b: 40, l: 55 },
    legend: { orientation: 'h' },
    hovermode: 'x unified',
  };
}

function drawPanels(traces, visibility) {
  const filtered = filterTraces(traces, visibility);
  const plots = {};

  PANEL_CONFIG.forEach(({ key, elementId, title, yTitle }) => {
    const panelTraces = filtered.filter((trace) => trace.meta.panel === key);
    const element = document.getElementById(elementId);
    console.log(`[panel=${key}] elementId=${elementId} element=`, element, `traces=${panelTraces.length}`);
    if (panelTraces.length === 0) {
      Plotly.react(
        element,
        [],
        {
          ...baseLayout(title, yTitle),
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
    Plotly.react(element, panelTraces, baseLayout(title, yTitle), { responsive: true });
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
