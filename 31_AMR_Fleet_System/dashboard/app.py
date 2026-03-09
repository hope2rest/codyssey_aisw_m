"""app.py - Fleet 모니터링 웹 대시보드.

Flask 기반 단일 페이지 대시보드로 5대 AMR의 실시간 상태,
KPI 지표, 작업 대기열을 시각화한다.
ROS2 없이 독립 실행 가능 (Demo 모드).

실행:
    python dashboard/app.py          # http://localhost:8080
    python dashboard/app.py --port 9090
"""

import sys
import os
import json
import time
import math
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─── 시뮬레이션 데이터 (Demo 모드) ─────────────────────────────

class FleetSimulator:
    """ROS2 없이 Fleet 상태를 시뮬레이션하는 데모 데이터 생성기."""

    WAREHOUSE_W, WAREHOUSE_H = 60.0, 40.0

    # 작업 위치 사전 정의
    LOCATIONS = {
        'Dock-1': (2, 35), 'Dock-2': (2, 30),
        'Dock-A': (58, 35), 'Dock-B': (58, 30),
        'Rack-A1': (8, 25), 'Rack-A3': (20, 25), 'Rack-A5': (32, 25),
        'Rack-B2': (14, 18), 'Rack-B4': (26, 18), 'Rack-B6': (38, 18),
        'Rack-C1': (8, 11), 'Rack-C3': (20, 11), 'Rack-C5': (32, 11),
        'Charging': (8, 3), 'Waiting': (50, 3),
    }

    def __init__(self):
        self.robots = {}
        self.tasks = []
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_counter = 0

        robot_starts = [
            (10, 20), (20, 15), (30, 25), (40, 10), (50, 20)]
        colors = ['#FF4444', '#4488FF', '#44CC44', '#FFCC00', '#CC44FF']

        for i in range(5):
            rid = f'amr_{i+1:02d}'
            self.robots[rid] = {
                'id': rid,
                'position': list(robot_starts[i]) + [0.0],
                'target': None,
                'battery': random.uniform(60, 100),
                'state': 'idle',
                'current_task': None,
                'color': colors[i],
                'trail': [list(robot_starts[i])],
                'speed': 0.0,
            }

        # 초기 작업 생성
        for _ in range(8):
            self._generate_task()

    def _generate_task(self):
        locs = list(self.LOCATIONS.keys())
        src = random.choice(locs[:4])   # Dock
        dst = random.choice(locs[4:13]) # Rack
        self.task_counter += 1
        self.tasks.append({
            'id': f'TASK-{self.task_counter:04d}',
            'from': src,
            'to': dst,
            'from_position': list(self.LOCATIONS[src]),
            'to_position': list(self.LOCATIONS[dst]),
            'priority': random.choice(['high', 'medium', 'low']),
            'status': 'pending',
            'assigned_robot': None,
            'created_at': time.time(),
        })

    def update(self):
        """1 프레임 업데이트."""
        for rid, robot in self.robots.items():
            if robot['state'] == 'idle':
                # 대기 중인 작업 할당
                for task in self.tasks:
                    if task['status'] == 'pending':
                        task['status'] = 'in_progress'
                        task['assigned_robot'] = rid
                        robot['state'] = 'navigating'
                        robot['current_task'] = task['id']
                        robot['target'] = task['to_position']
                        break

            elif robot['state'] == 'navigating' and robot['target']:
                # 목표를 향해 이동
                dx = robot['target'][0] - robot['position'][0]
                dy = robot['target'][1] - robot['position'][1]
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < 0.5:
                    # 도착
                    robot['position'][0] = robot['target'][0]
                    robot['position'][1] = robot['target'][1]
                    robot['state'] = 'docking'
                    robot['speed'] = 0.0
                else:
                    speed = min(1.5, dist * 0.5)
                    robot['position'][0] += (dx / dist) * speed * 0.3
                    robot['position'][1] += (dy / dist) * speed * 0.3
                    robot['position'][2] = math.atan2(dy, dx)
                    robot['speed'] = speed
                    robot['battery'] = max(5, robot['battery'] - 0.01)

                robot['trail'].append(
                    [robot['position'][0], robot['position'][1]])
                if len(robot['trail']) > 100:
                    robot['trail'] = robot['trail'][-100:]

            elif robot['state'] == 'docking':
                # 도킹 완료 → 작업 완료
                for task in self.tasks:
                    if task['id'] == robot['current_task']:
                        task['status'] = 'completed'
                        break
                robot['state'] = 'idle'
                robot['current_task'] = None
                robot['target'] = None
                self.completed_tasks += 1

                # 새 작업 생성
                if random.random() < 0.6:
                    self._generate_task()

    def get_robots(self):
        return list(self.robots.values())

    def get_kpi(self):
        elapsed = max(1, time.time() - self.start_time)
        hours = elapsed / 3600.0
        active = sum(
            1 for r in self.robots.values() if r['state'] != 'idle')
        utilization = active / len(self.robots) * 100

        return {
            'tasks_per_hour': round(
                self.completed_tasks / max(0.01, hours), 1),
            'utilization': round(utilization, 1),
            'active_robots': active,
            'total_robots': len(self.robots),
            'deadlocks': 0,
            'completed_tasks': self.completed_tasks,
            'pending_tasks': sum(
                1 for t in self.tasks if t['status'] == 'pending'),
            'elapsed_minutes': round(elapsed / 60, 1),
        }

    def get_tasks(self):
        # 최근 20개
        return sorted(
            self.tasks, key=lambda t: t['created_at'], reverse=True)[:20]

    def get_map_data(self):
        """물류센터 레이아웃 데이터."""
        shelves = []
        for name, pos in self.LOCATIONS.items():
            if name.startswith('Rack'):
                shelves.append({
                    'name': name, 'x': pos[0], 'y': pos[1],
                    'w': 4, 'h': 2})
        return {
            'width': self.WAREHOUSE_W,
            'height': self.WAREHOUSE_H,
            'shelves': shelves,
            'docks': [
                {'name': k, 'x': v[0], 'y': v[1]}
                for k, v in self.LOCATIONS.items()
                if k.startswith('Dock')],
            'charging': self.LOCATIONS['Charging'],
        }


# ─── HTTP 서버 ──────────────────────────────────────────────

simulator = FleetSimulator()


def _update_loop():
    """백그라운드에서 시뮬레이션 업데이트."""
    while True:
        simulator.update()
        time.sleep(0.3)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AMR Fleet Dashboard</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#1a1a2e; color:#e0e0e0; font-family:'Segoe UI',sans-serif; }
.header { background:#16213e; padding:12px 24px; display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #0f3460; }
.header h1 { font-size:20px; color:#4cc9f0; }
.header .status { font-size:13px; color:#8d99ae; }
.grid { display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; gap:12px; padding:12px; height:calc(100vh - 56px); }
.panel { background:#16213e; border-radius:8px; padding:16px; border:1px solid #0f3460; overflow:hidden; display:flex; flex-direction:column; }
.panel h2 { font-size:14px; color:#4cc9f0; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px; }
canvas { width:100%; flex:1; border-radius:4px; background:#0a0a1a; }
.kpi-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; flex:1; }
.kpi-card { background:#0f3460; border-radius:8px; padding:16px; text-align:center; display:flex; flex-direction:column; justify-content:center; }
.kpi-card .value { font-size:32px; font-weight:bold; color:#4cc9f0; }
.kpi-card .label { font-size:12px; color:#8d99ae; margin-top:4px; }
.kpi-card.alert .value { color:#ff6b6b; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { background:#0f3460; padding:8px; text-align:left; color:#4cc9f0; position:sticky; top:0; }
td { padding:6px 8px; border-bottom:1px solid #1a1a3e; }
.badge { padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold; }
.badge.completed { background:#2d6a4f; color:#95d5b2; }
.badge.in_progress { background:#7b5e00; color:#ffd166; }
.badge.pending { background:#3d3d5c; color:#8d99ae; }
.badge.high { background:#6a040f; color:#ff758f; }
.badge.medium { background:#7b5e00; color:#ffd166; }
.badge.low { background:#2d4a3e; color:#95d5b2; }
.task-table-wrap { flex:1; overflow-y:auto; }
.chart-wrap { flex:1; position:relative; }
</style>
</head>
<body>
<div class="header">
  <h1>AMR Fleet Monitoring Dashboard</h1>
  <div class="status" id="clock">--</div>
</div>
<div class="grid">
  <div class="panel">
    <h2>Warehouse Map</h2>
    <canvas id="mapCanvas"></canvas>
  </div>
  <div class="panel">
    <h2>KPI</h2>
    <div class="kpi-grid">
      <div class="kpi-card"><div class="value" id="kpi-tph">--</div><div class="label">Tasks / Hour</div></div>
      <div class="kpi-card"><div class="value" id="kpi-util">--</div><div class="label">Utilization %</div></div>
      <div class="kpi-card"><div class="value" id="kpi-active">--</div><div class="label">Active Robots</div></div>
      <div class="kpi-card" id="kpi-dl-card"><div class="value" id="kpi-dl">--</div><div class="label">Deadlocks</div></div>
    </div>
  </div>
  <div class="panel">
    <h2>Tasks Completed</h2>
    <div class="chart-wrap"><canvas id="chartCanvas"></canvas></div>
  </div>
  <div class="panel">
    <h2>Task Queue</h2>
    <div class="task-table-wrap">
      <table><thead><tr><th>Status</th><th>Task ID</th><th>Robot</th><th>From</th><th>To</th><th>Priority</th></tr></thead>
      <tbody id="taskBody"></tbody></table>
    </div>
  </div>
</div>
<script>
const mapCanvas = document.getElementById('mapCanvas');
const chartCanvas = document.getElementById('chartCanvas');
const mapCtx = mapCanvas.getContext('2d');
const chartCtx = chartCanvas.getContext('2d');
let chartData = [];

function resize() {
  [mapCanvas, chartCanvas].forEach(c => {
    c.width = c.clientWidth * devicePixelRatio;
    c.height = c.clientHeight * devicePixelRatio;
    c.getContext('2d').scale(devicePixelRatio, devicePixelRatio);
  });
}
window.addEventListener('resize', resize);
resize();

function drawMap(robots, mapData) {
  const ctx = mapCtx;
  const W = mapCanvas.clientWidth, H = mapCanvas.clientHeight;
  const sx = W / mapData.width, sy = H / mapData.height;
  ctx.clearRect(0, 0, W, H);

  // 바닥
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  // 그리드
  ctx.strokeStyle = '#252545';
  ctx.lineWidth = 0.5;
  for (let x = 0; x < mapData.width; x += 5) {
    ctx.beginPath(); ctx.moveTo(x*sx, 0); ctx.lineTo(x*sx, H); ctx.stroke();
  }
  for (let y = 0; y < mapData.height; y += 5) {
    ctx.beginPath(); ctx.moveTo(0, y*sy); ctx.lineTo(W, y*sy); ctx.stroke();
  }

  // 선반
  ctx.fillStyle = '#2d3a5c';
  ctx.strokeStyle = '#4a5a8c';
  ctx.lineWidth = 1;
  (mapData.shelves || []).forEach(s => {
    const rx = s.x * sx, ry = (mapData.height - s.y) * sy;
    ctx.fillRect(rx - s.w*sx/2, ry - s.h*sy/2, s.w*sx, s.h*sy);
    ctx.strokeRect(rx - s.w*sx/2, ry - s.h*sy/2, s.w*sx, s.h*sy);
    ctx.fillStyle = '#5a6a9c'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(s.name, rx, ry + 3);
    ctx.fillStyle = '#2d3a5c';
  });

  // 도크
  (mapData.docks || []).forEach(d => {
    const dx = d.x * sx, dy = (mapData.height - d.y) * sy;
    ctx.fillStyle = '#0f3460'; ctx.fillRect(dx-12, dy-8, 24, 16);
    ctx.fillStyle = '#4cc9f0'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(d.name, dx, dy+3);
  });

  // 로봇 궤적 + 위치
  robots.forEach(r => {
    const trail = r.trail || [];
    if (trail.length > 1) {
      ctx.strokeStyle = r.color + '44';
      ctx.lineWidth = 2;
      ctx.beginPath();
      trail.forEach((p, i) => {
        const px = p[0]*sx, py = (mapData.height - p[1])*sy;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    const rx = r.position[0]*sx, ry = (mapData.height - r.position[1])*sy;
    // 로봇 본체
    ctx.fillStyle = r.color;
    ctx.beginPath(); ctx.arc(rx, ry, 7, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 8px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(r.id.slice(-2), rx, ry+3);
    // 방향
    const angle = r.position[2] || 0;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(rx, ry);
    ctx.lineTo(rx + Math.cos(-angle)*12, ry + Math.sin(-angle)*12);
    ctx.stroke();
  });
}

function drawChart() {
  const ctx = chartCtx;
  const W = chartCanvas.clientWidth, H = chartCanvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  if (chartData.length < 2) return;

  const maxVal = Math.max(...chartData.map(d=>d.v), 1);
  const padL = 40, padB = 24, padT = 10, padR = 10;
  const cw = W - padL - padR, ch = H - padT - padB;

  // 축
  ctx.strokeStyle = '#3d3d5c'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, H-padB); ctx.lineTo(W-padR, H-padB); ctx.stroke();

  // 눈금
  ctx.fillStyle = '#8d99ae'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = Math.round(maxVal * i / 4);
    const y = H - padB - (ch * i / 4);
    ctx.fillText(val, padL - 4, y + 3);
    ctx.strokeStyle = '#252545';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W-padR, y); ctx.stroke();
  }

  // 라인
  ctx.strokeStyle = '#4cc9f0'; ctx.lineWidth = 2;
  ctx.beginPath();
  chartData.forEach((d, i) => {
    const x = padL + (i / (chartData.length-1)) * cw;
    const y = H - padB - (d.v / maxVal) * ch;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 영역 채우기
  const grad = ctx.createLinearGradient(0, padT, 0, H-padB);
  grad.addColorStop(0, 'rgba(76,201,240,0.3)');
  grad.addColorStop(1, 'rgba(76,201,240,0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  chartData.forEach((d, i) => {
    const x = padL + (i / (chartData.length-1)) * cw;
    const y = H - padB - (d.v / maxVal) * ch;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(padL + cw, H-padB);
  ctx.lineTo(padL, H-padB);
  ctx.closePath(); ctx.fill();
}

function updateTasks(tasks) {
  const tbody = document.getElementById('taskBody');
  tbody.innerHTML = tasks.map(t => `<tr>
    <td><span class="badge ${t.status}">${t.status.replace('_',' ')}</span></td>
    <td>${t.id}</td>
    <td>${t.assigned_robot || '-'}</td>
    <td>${t.from}</td>
    <td>${t.to}</td>
    <td><span class="badge ${t.priority}">${t.priority}</span></td>
  </tr>`).join('');
}

async function refresh() {
  try {
    const [rRes, kRes, tRes, mRes] = await Promise.all([
      fetch('/api/robots'), fetch('/api/kpi'),
      fetch('/api/tasks'), fetch('/api/map')
    ]);
    const robots = await rRes.json();
    const kpi = await kRes.json();
    const tasks = await tRes.json();
    const mapData = await mRes.json();

    drawMap(robots, mapData);
    updateTasks(tasks);

    document.getElementById('kpi-tph').textContent = kpi.tasks_per_hour;
    document.getElementById('kpi-util').textContent = kpi.utilization + '%';
    document.getElementById('kpi-active').textContent = kpi.active_robots + '/' + kpi.total_robots;
    document.getElementById('kpi-dl').textContent = kpi.deadlocks;
    document.getElementById('kpi-dl-card').className = 'kpi-card' + (kpi.deadlocks > 0 ? ' alert' : '');

    chartData.push({v: kpi.completed_tasks});
    if (chartData.length > 60) chartData.shift();
    drawChart();

    document.getElementById('clock').textContent =
      new Date().toLocaleTimeString('ko-KR') + ' | ' +
      kpi.elapsed_minutes + 'min | ' + kpi.completed_tasks + ' tasks done';
  } catch(e) { console.error(e); }
}

setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP 요청 핸들러."""

    def log_message(self, format, *args):
        pass  # 로그 억제

    def _respond_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(
            data, ensure_ascii=False).encode('utf-8'))

    def _respond_html(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self._respond_html(DASHBOARD_HTML)
        elif path == '/api/robots':
            self._respond_json(simulator.get_robots())
        elif path == '/api/kpi':
            self._respond_json(simulator.get_kpi())
        elif path == '/api/tasks':
            self._respond_json(simulator.get_tasks())
        elif path == '/api/map':
            self._respond_json(simulator.get_map_data())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = 8080
    if '--port' in sys.argv:
        idx = sys.argv.index('--port')
        port = int(sys.argv[idx + 1])

    # 시뮬레이션 백그라운드 스레드
    t = threading.Thread(target=_update_loop, daemon=True)
    t.start()

    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f'AMR Fleet Dashboard: http://localhost:{port}')
    print('Ctrl+C로 종료')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n종료')
        server.server_close()


if __name__ == '__main__':
    main()
