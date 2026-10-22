"""Write the acceptance dashboard page (files/index.html text) from sections.json:
one card per section, its table marked for the section's dataset."""
import html
import json
import sys

secs = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
LOWER_IS_BETTER = {"summary": ["stand_hip"], "hip": ["flat", "run", "up", "down"]}
esc = lambda s: html.escape(s, quote=True)

cards = []
nav = []
for s in secs:
  sid = f"sec-{s['id']}"
  nav.append(f'<a class="acc-navlink" href="#{sid}">{esc(s["title"])}</a>')
  ths = "".join(f'<th scope="col" data-field="{esc(f)}">{esc(h)}</th>' for f, h in s["cols"])
  cards.append(f'''  <section class="acc-card" id="{sid}">
    <h2 class="acc-h2">{esc(s["title"])}</h2>
    <p class="acc-note">{esc(s["note"])}</p>
    <div class="acc-scroll">
      <table class="acc-table" data-source="{esc(s["id"])}" data-row-key="{esc(s.get("key", "model"))}">
        <thead><tr>{ths}</tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>''')

page = f'''<style>
.acc {{ font-family: var(--font-anthropic-sans); color: var(--color-fg); display: grid; gap: var(--cds-gap-lg); }}
.acc-head {{ display: grid; gap: var(--cds-gap-sm); }}
.acc-title {{ font-family: var(--font-anthropic-serif); font-weight: 400; font-size: var(--cds-font-size-title); margin: 0; text-wrap: balance; }}
.acc-sub {{ margin: 0; color: var(--color-fg-muted); font-size: var(--cds-font-size-body); max-width: 70ch; }}
.acc-bar {{ display: flex; flex-wrap: wrap; gap: var(--cds-gap-xs); align-items: center; }}
.acc-navlink {{ font-size: var(--cds-font-size-caption); color: var(--color-fg); text-decoration: none; border: 1px solid var(--color-border-line); border-radius: 999px; padding: 2px 10px; background: var(--color-panel); }}
.acc-navlink:hover, .acc-navlink:focus-visible {{ border-color: var(--color-fg-muted); }}
.acc-filter {{ display: flex; flex-wrap: wrap; gap: var(--cds-gap-xs); }}
.acc-filter button {{ font: inherit; font-size: var(--cds-font-size-caption); color: var(--color-fg); background: transparent; border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: 3px 10px; cursor: pointer; }}
.acc-filter button[aria-pressed="true"] {{ background: var(--color-fg); color: var(--color-bg); border-color: var(--color-fg); }}
.acc-filter button:focus-visible {{ outline: 2px solid var(--color-fg-muted); outline-offset: 1px; }}
.acc-label {{ font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); margin-right: var(--cds-gap-xs); }}
.acc-card {{ background: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: var(--cds-pad-lg); display: grid; gap: var(--cds-gap-sm); min-width: 0; scroll-margin-top: 12px; }}
.acc-h2 {{ margin: 0; font-size: var(--cds-font-size-heading); font-weight: var(--cds-font-weight-medium); }}
.acc-note {{ margin: 0; color: var(--color-fg-muted); font-size: var(--cds-font-size-caption); max-width: 90ch; }}
.acc-scroll {{ overflow-x: auto; min-width: 0; }}
.acc-table {{ border-collapse: collapse; width: 100%; font-size: var(--cds-font-size-caption); font-variant-numeric: tabular-nums; }}
.acc-table th {{ text-align: left; font-weight: var(--cds-font-weight-medium); color: var(--color-fg-muted); padding: 6px 10px; border-bottom: 1px solid var(--color-border-line); white-space: nowrap; vertical-align: bottom; }}
.acc-table td {{ padding: 5px 10px; border-bottom: 1px solid var(--color-border-line); white-space: nowrap; vertical-align: top; }}
.acc-table td:first-child, .acc-table th:first-child {{ position: sticky; left: 0; background: var(--color-panel); z-index: 1; }}
.acc-table td:first-child {{ font-weight: var(--cds-font-weight-medium); }}
.acc-table tr.acc-first td {{ border-top: 2px solid var(--color-border-line); }}
.acc-table tr.acc-deployed td {{ color: var(--color-fg-muted); }}
.acc-table tr.acc-off {{ display: none; }}
.acc-table td.acc-wrap {{ white-space: normal; min-width: 22rem; }}
.acc-table td.acc-wrap-s {{ white-space: normal; min-width: 6rem; }}
.acc-table td.acc-wrap-s .acc-chip {{ margin: 1px 4px 1px 0; }}
.acc-fail {{ color: var(--color-bad); }}
.acc-chip {{ display: inline-block; padding: 0 8px; border-radius: 999px; border: 1px solid currentColor; line-height: 1.6; }}
.acc-chip.ok {{ color: var(--color-ok); }}
.acc-chip.bad {{ color: var(--color-bad); font-weight: var(--cds-font-weight-medium); }}
.acc-muted {{ color: var(--color-fg-muted); }}
.acc-delta.ok {{ color: var(--color-ok); }}
.acc-delta.bad {{ color: var(--color-bad); }}
.acc [data-dash-source-state=refreshing] {{ opacity: .5; transition: opacity var(--cds-dur-slow); }}
</style>
<div class="acc">
  <header class="acc-head">
    <h1 class="acc-title">盲走策略验收</h1>
    <p class="acc-sub">六台机型同一套验收：每台列现部署模型、本轮起点模型和本轮的新存档。三项门槛判通过或不通过，其余数据对比现部署。轮式比腿式多一项前进后退四轮着地。</p>
    <nav class="acc-bar" aria-label="章节">{"".join(nav)}</nav>
    <div class="acc-bar"><span class="acc-label">机型</span><div class="acc-filter" role="group" aria-label="按机型筛选">
      <button type="button" data-robot="all" aria-pressed="true">全部</button>
      <button type="button" data-robot="Go2" aria-pressed="false">Go2</button>
      <button type="button" data-robot="As2" aria-pressed="false">As2</button>
      <button type="button" data-robot="B2" aria-pressed="false">B2</button>
      <button type="button" data-robot="Go2W" aria-pressed="false">Go2W</button>
      <button type="button" data-robot="As2W" aria-pressed="false">As2W</button>
      <button type="button" data-robot="B2W" aria-pressed="false">B2W</button>
    </div></div>
  </header>
{chr(10).join(cards)}
</div>
<script>
(function () {{
  var root = document.querySelector('.acc');
  var tables = Array.prototype.slice.call(root.querySelectorAll('table[data-source]'));
  var lower = {json.dumps(LOWER_IS_BETTER)};
  var robot = 'all';
  var drawn = false;
  function span(td, text, cls) {{
    var s = document.createElement('span');
    s.className = cls;
    s.textContent = text;
    td.appendChild(s);
  }}
  function cell(td, id, field, v) {{
    var s = v == null ? '—' : String(v);
    if (s === '通过') return span(td, s, 'acc-chip ok');
    if (s === '不通过') return span(td, s, 'acc-chip bad');
    if (s.indexOf('不通过：') === 0) {{
      s.slice(4).split('、').forEach(function (tag) {{ span(td, tag, 'acc-chip bad'); }});
      return;
    }}
    if (s === '评测中' || s === '不适用' || s === '未训练' || s === '未测' || s === '基准') return span(td, s, 'acc-muted');
    var colored = (lower[id] || []).indexOf(field) >= 0;
    var re = /\\(([+-]\\d+)%\\)/g, last = 0, m;
    while ((m = re.exec(s))) {{
      td.appendChild(document.createTextNode(s.slice(last, m.index)));
      var n = parseInt(m[1], 10);
      span(td, m[0], 'acc-delta' + (colored && n < 0 ? ' ok' : colored && n > 0 ? ' bad' : ''));
      last = m.index + m[0].length;
    }}
    td.appendChild(document.createTextNode(s.slice(last)));
  }}
  function fill(table) {{
    var id = table.getAttribute('data-source');
    var d = dash.data(id);
    var tbody = table.tBodies[0];
    var fields = Array.prototype.map.call(table.tHead.rows[0].cells, function (th) {{ return th.getAttribute('data-field'); }});
    tbody.textContent = '';
    if (d.status !== 'ok') {{
      var tr = document.createElement('tr');
      var td = document.createElement('td');
      td.colSpan = fields.length;
      span(td, d.status === 'error' ? '加载失败' : '—', 'acc-muted');
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }}
    var prev = null;
    d.data.forEach(function (row) {{
      var tr = document.createElement('tr');
      tr.setAttribute('data-key', row[table.getAttribute('data-row-key')]);
      if (row.robot) {{
        tr.setAttribute('data-robot', row.robot);
        tr.className = 'acc-' + row.kind + (row.robot !== prev ? ' acc-first' : '');
        prev = row.robot;
      }}
      fields.forEach(function (f) {{
        var td = document.createElement('td');
        if (f === 'fails' || f === 'meaning') td.className = 'acc-wrap';
        else if (f.indexOf('g_') === 0) td.className = 'acc-wrap-s';
        cell(td, id, f, row[f]);
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
  }}
  function applyFilter() {{
    tables.forEach(function (t) {{
      Array.prototype.forEach.call(t.tBodies[0].rows, function (tr) {{
        var r = tr.getAttribute('data-robot');
        tr.classList.toggle('acc-off', robot !== 'all' && r !== null && r !== robot);
      }});
    }});
  }}
  Array.prototype.forEach.call(root.querySelectorAll('.acc-filter button'), function (b) {{
    b.addEventListener('click', function () {{
      robot = b.getAttribute('data-robot');
      Array.prototype.forEach.call(root.querySelectorAll('.acc-filter button'), function (o) {{
        o.setAttribute('aria-pressed', o === b ? 'true' : 'false');
      }});
      applyFilter();
    }});
  }});
  Array.prototype.forEach.call(root.querySelectorAll('.acc-navlink'), function (a) {{
    a.addEventListener('click', function (e) {{
      var el = document.getElementById(a.getAttribute('href').slice(1));
      if (!el) return;
      e.preventDefault();
      el.scrollIntoView();
      dash.setLink(el.id);
    }});
  }});
  dash.onData(function () {{
    tables.forEach(fill);
    applyFilter();
    if (!drawn) {{
      drawn = true;
      var named = location.hash ? document.getElementById(location.hash.slice(1)) : null;
      if (named) {{ named.scrollIntoView(); dash.setLink(named.id); }}
    }}
  }});
}})();
</script>
'''
open(out_path, "w").write(page)
print(len(page.encode()), "bytes")
